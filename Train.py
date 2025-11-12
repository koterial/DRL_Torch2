import time
import queue
import numpy as np
import gymnasium as gym
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from Core.Agent.Agent_Factory import Agent, Factory
from Core.Agent.TD3.TD3_Roles import TD3_Learner, TD3_Collector
from Core.Buffer.Buffer_Create import buffer_create
from Core.Noise.Noise_Create import noise_create
from Env.Env_Adapter import env_adapter


def run_agent_main(env_name, agent_config, buffer_config, max_episodes=10000, max_steps=1600, warmup_steps=10000, log_dir_root="Logs/Single Process"):
    print("--- 启动模式: 单进程 (Agent) ---")
    if SummaryWriter and log_dir_root != None:
        log_dir = f"{log_dir_root}/{env_name}_{int(time.time())}"
        writer = SummaryWriter(log_dir)
        print(f"[Agent] TensorBoard 日志保存至: {log_dir}")
    else:
        writer = None

    env = gym.make(env_name)
    action_bound = float(env.action_space.high[0])
    agent_config["state_shape"] = [env.observation_space.shape[0]]
    agent_config["action_shape"] = [env.action_space.shape[0]]

    agent_facade_config = {
        "learner_class": TD3_Learner,
        "collector_class": TD3_Collector,
        "agent_config": agent_config,
        "buffer_config": buffer_config,
    }
    agent = Agent(**agent_facade_config)

    explore_noise_config = {
        "index": "collector_0",
        "class": "Gaussian",
        "action_shape": agent_config["action_shape"],
        "scale": 0.5,
        "bound": 0.5,
    }
    exploration_noise = noise_create(explore_noise_config)

    episode_reward_list = []
    total_steps = 0
    start_time = time.time()
    print(f"开始训练, 热启动 {warmup_steps} 步")
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            total_steps += 1
            if total_steps < warmup_steps:
                action = env.action_space.sample()
                log_prob = None
            else:
                action, log_prob = agent.get_action(state)
                noise = exploration_noise.get_noise()
                action = np.clip(action + noise, -agent.collector.action_bound, agent.collector.action_bound)
            act = action * action_bound
            next_state, reward, terminated, truncated, _ = env.step(act)
            state, action, log_prob, next_state, reward, terminated, truncated = env_adapter(env_name, state, action, log_prob, next_state, reward, terminated, truncated)
            agent.remember((state, action, log_prob, next_state, reward, terminated, truncated))
            state = next_state
            episode_reward += reward
            exploration_noise.bound_decay()
            if total_steps > warmup_steps:
                agent.train()
            if terminated or truncated:
                break
        episode_reward_list.append(episode_reward)
        print(f"Episode: {episode}, Steps: {step}, Reward: {episode_reward:.2f}, Actor Loss: {agent.learner.actor.loss:.4f}, Critic Loss: {agent.learner.critic.loss:.4f}")
        if writer:
            writer.add_scalar("Episode/Step", step, episode)
            writer.add_scalar("Episode/Reward", episode_reward, episode)
            writer.add_scalar("Episode/Noise", exploration_noise.bound, episode)
            writer.add_scalar("Train/Actor Loss", agent.learner.actor.loss, episode)
            writer.add_scalar("Train/Critic Loss", agent.learner.critic.loss, episode)
    end_time = time.time()
    print(f"训练完成, 总用时: {(end_time - start_time) / 60:.2f} 分钟")
    env.close()
    if writer:
        writer.close()


def run_factory_main(env_name, agent_config, buffer_config, num_collectors=4, num_learners=1, log_dir_root="Logs/Multi Process"):
    print("--- 启动模式: 多进程 (Factory) ---")
    mp.set_start_method('spawn')
    if SummaryWriter and log_dir_root != None:
        log_dir = f"{log_dir_root}/{env_name}_{int(time.time())}"
        print(f"[Factory] TensorBoard 日志保存至: {log_dir}")
    else:
        log_dir = None

    factory_config = {
        "learner_class": TD3_Learner,
        "collector_class": TD3_Collector,
        "num_learners": num_learners,
        "num_collectors": num_collectors,
        "agent_config": agent_config,
        "buffer_config": buffer_config,
    }
    factory = Factory(**factory_config)
    print("Factory 创建成功, 准备启动进程")

    buffer_config = factory.buffer_config
    buffer_config["state_shape"] = agent_config["state_shape"]
    buffer_config["action_shape"] = agent_config["action_shape"]
    replay_buffer = buffer_create(buffer_config)
    prioritized_replay = True if buffer_config.get("class") == "Prioritized_Replay_Buffer" else False
    batch_shape = factory.agent_config.get("batch_shape", 256)
    experience_queue = factory.get_experience_queue()
    sample_queue = factory.get_sample_queue()
    error_queue = factory.get_error_queue()

    processes = []
    try:
        for i in range(factory.num_learners):
            learner_conf = factory.get_learner_config(i)
            p_learner = mp.Process(target=run_learner_process, args=(factory.get_learner_class(), learner_conf, log_dir))
            p_learner.start()
            processes.append(p_learner)
        for i in range(factory.num_collectors):
            collector_conf = factory.get_collector_config(i)
            p_collector = mp.Process(target=run_collector_process, args=(factory.get_collector_class(), collector_conf, env_name, log_dir))
            p_collector.start()
            processes.append(p_collector)
        print(f"已启动 {len(processes)} 个进程. ({factory.num_learners} Learner, {factory.num_collectors} Collectors)")
        print("按 [Ctrl+C] 停止训练.")
        while True:
            try:
                experience_batch = experience_queue.get_nowait()
                replay_buffer.remember(experience_batch)
            except queue.Empty:
                pass
            if replay_buffer.size() > batch_shape and not sample_queue.full():
                batch = replay_buffer.sample(batch_shape)
                if batch[0] is not None:
                    sample_queue.put(batch)
            if prioritized_replay and not error_queue.empty():
                try:
                    index_batch, error_batch = error_queue.get_nowait()
                    replay_buffer.batch_update(index_batch, error_batch)
                except queue.Empty:
                    pass
            if experience_queue.empty() and sample_queue.full():
                time.sleep(0.01)
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("\n收到 [Ctrl+C] 信号... 正在终止所有进程...")
    except Exception as e:
        print(f"主进程发生致命错误: {e}")
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        print("所有进程已终止.")


def run_learner_process(learner_class, learner_config, log_dir, log_freq=100):
    learner = learner_class(**learner_config)
    if SummaryWriter and log_dir != None:
        writer = SummaryWriter(f"{log_dir}/Learner_{learner.index}")
        print(f"[{learner.index}] TensorBoard 日志保存至: {log_dir}/Learner_{learner.index}")
    else:
        writer = None
    print(f"[{learner.index}] 启动在 {learner.device}")
    learner._distribute_weights()
    print(f"[{learner.index}] 已分发初始权重")

    while True:
        try:
            learner.train()
            if learner.step % log_freq == 0:
                print(f"[{learner.index}] Step: {learner.step}, Actor Loss: {learner.actor.loss:.4f}, Critic Loss: {learner.critic.loss:.4f}")
            if writer:
                writer.add_scalar("Train/Actor Loss", learner.actor.loss, learner.step)
                writer.add_scalar("Train/Critic Loss", learner.critic.loss, learner.step)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[{learner.index}] 发生致命错误: {e}")
            break

    if writer:
        writer.close()
    print(f"[{learner.index}] 终止.")


def run_collector_process(collector_class, collector_config, env_name, log_dir, log_freq=100):
    collector = collector_class(**collector_config)
    if SummaryWriter and log_dir != None:
        writer = SummaryWriter(f"{log_dir}/Collector_{collector.index}")
        print(f"[{collector.index}] TensorBoard 日志保存至: {log_dir}/Collector_{collector.index}")
    else:
        writer = None

    try:
        env = gym.make(env_name)
        action_bound = float(env.action_space.high[0])
        explore_noise_config = {
            "index": "collector_0",
            "class": "Gaussian",
            "action_shape": collector_config["action_shape"],
            "scale": 0.5,
            "bound": 0.5,
        }
        exploration_noise = noise_create(explore_noise_config)
    except Exception as e:
        print(f"[{collector.index}] 创建环境失败: {e}")
        return

    print(f"[{collector.index}] 启动在 {collector.device}")

    try:
        weight_queue = collector_config["weight_queue"]
        initial_weights = weight_queue.get(timeout=10)
        collector.set_actor_state_dict(initial_weights)
        print(f"[{collector.index}] 收到初始权重, 开始采集")
    except queue.Empty:
        print(f"[{collector.index}] 错误: 等待初始权重超时")
        return
    except Exception as e:
        print(f"[{collector.index}] 接收初始权重时出错: {e}")
        return

    state, _ = env.reset()
    episode_reward = 0
    total_episodes = 0
    total_steps = 0
    episode_step = 0
    while True:
        try:
            total_steps += 1
            episode_step += 1
            action, log_prob = collector.get_action(state)
            noise = exploration_noise.get_noise()
            action = np.clip(action + noise, -collector.action_bound, collector.action_bound)
            act = action * action_bound
            next_state, reward, terminated, truncated, _ = env.step(act)
            state, action, log_prob, next_state, reward, terminated, truncated = env_adapter(env_name, state, action, log_prob, next_state, reward, terminated, truncated)
            collector.remember((state, action, log_prob, next_state, reward, terminated, truncated))
            state = next_state
            episode_reward += reward
            exploration_noise.bound_decay()
            if terminated or truncated:
                if total_episodes % log_freq == 0:
                    print(f"[{collector.index}] Episode: {total_episodes}, Steps: {episode_step}, Reward: {episode_reward:.2f}")
                if writer:
                    writer.add_scalar("Episode/Step", episode_step, total_episodes)
                    writer.add_scalar("Episode/Reward", episode_reward, total_episodes)
                    writer.add_scalar("Episode/Noise", exploration_noise.bound, total_episodes)
                state, _ = env.reset()
                total_episodes += 1
                episode_step = 0
                episode_reward = 0

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[{collector.index}] 采集时发生错误: {e}")
            time.sleep(1)

    if writer:
        writer.close()
    env.close()
    print(f"[{collector.index}] 终止")


if __name__ == "__main__":
    multi_processing = True
    num_collectors = 8

    env_name = "BipedalWalker-v3"
    temp_env = gym.make(env_name)
    state_shape = [temp_env.observation_space.shape[0]]
    action_shape = [temp_env.action_space.shape[0]]
    temp_env.close()

    agent_config = {
        "state_shape": state_shape, "action_shape": action_shape,
        "actor_lr": 1e-3, "critic_lr": 1e-3, "batch_shape": 512 * num_collectors,
        "reward_gamma": 0.99, "actor_train_freq": 2, "update_freq": 2, "update_tau": 0.005,
        "eval_noise_std": 0.2, "eval_noise_bound": 0.5,
        "actor_hidden_shape": [400, 300], "critic_hidden_shape": [400, 300], "hidden_activation": "relu",
        "learner_device": "cuda" if torch.cuda.is_available() else "cpu", "collector_device": "cpu",
    }

    buffer_config = {
        "class": "Replay_Buffer",
        "shape": 1e6,
    }

    if multi_processing:
        run_factory_main(
            env_name=env_name,
            agent_config=agent_config,
            buffer_config=buffer_config,
            num_collectors=num_collectors,
        )

    else:
        run_agent_main(
            env_name=env_name,
            agent_config=agent_config,
            buffer_config=buffer_config,
            warmup_steps=10000,
        )