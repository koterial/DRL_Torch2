import copy
import queue
import torch
from Core.Agent.Actor_Critic.Actor import Deterministic_Actor
from Core.Agent.TD3.TD3_Components import TD3_Actor, TD3_Critic
from Core.Agent.Utils.Common import update_target_model

torch.set_default_dtype(torch.float32)


class TD3_Learner():
    def __init__(self, *args, **kwargs):
        #  --- 1. 弹出所有非简单类型的对象 ---
        self.all_weight_queues = kwargs.pop("all_weight_queues")
        self.sample_queue = kwargs.pop("sample_queue")
        self.error_queue = kwargs.pop("error_queue")

        self.config = copy.deepcopy(kwargs)

        # --- 2. 获取核心配置 ---
        self.index = self.config["index"]
        self.state_shape = self.config["state_shape"]
        self.action_shape = self.config["action_shape"]
        self.device = torch.device(self.config.get("device", "cuda"))

        # --- 3. 获取算法超参数 ---
        self.actor_lr = self.config.get("actor_lr", 1e-3)
        self.critic_lr = self.config.get("critic_lr", 1e-3)
        self.actor_train_freq = self.config.get("actor_train_freq", 2)
        self.reward_gamma = self.config.get("reward_gamma", 0.99)
        self.update_freq = self.config.get("update_freq", 2)
        self.update_tau = self.config.get("update_tau", 0.005)
        self.clip_norm = self.config.get("clip_norm", 0.5)
        self.batch_shape = int(self.config.get("batch_shape", 256))

        self.actor_hidden_shape = self.config.get("actor_hidden_shape", [256, 256])
        self.critic_hidden_shape = self.config.get("critic_hidden_shape", [256, 256])
        self.actor_activation = self.config.get("actor_activation", "tanh")
        self.critic_activation = self.config.get("critic_activation", "linear")
        self.hidden_activation = self.config.get("hidden_activation", "relu")

        self.action_bound = self.config.get("action_bound", 1.0)
        self.eval_noise_std = self.config.get("eval_noise_std", 0.2)
        self.eval_noise_bound = self.config.get("eval_noise_bound", 0.2)
        self.eval_noise_decay = self.config.get("eval_noise_decay", 0.999)
        self.step = 0

        # --- 4. 实例化训练Critic (GPU) ---
        critic_kwargs = {
            "state_shape": self.state_shape, "action_shape": self.action_shape,
            "hidden_shape": self.critic_hidden_shape, "activation": self.critic_activation,
            "hidden_activation": self.hidden_activation, "lr": self.critic_lr,
            "clip_norm": self.clip_norm,
        }
        self.critic = TD3_Critic(**critic_kwargs).to(self.device)
        self.critic_target = TD3_Critic(**critic_kwargs).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # --- 5. 实例化训练Actor (GPU) ---
        actor_kwargs = {
            "state_shape": self.state_shape, "action_shape": self.action_shape,
            "hidden_shape": self.actor_hidden_shape, "activation": self.actor_activation,
            "hidden_activation": self.hidden_activation, "lr": self.actor_lr,
            "clip_norm": self.clip_norm, "critic": self.critic
        }
        self.actor = TD3_Actor(**actor_kwargs).to(self.device)
        self.actor_target = TD3_Actor(**actor_kwargs).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

    # 将当前Actor参数分发给Collector
    def _distribute_weights(self):
        weights = self.get_actor_state_dict(self.device)
        for weight_queue in self.all_weight_queues:
            try:
                weight_queue.get_nowait()
            except:
                pass
            try:
                weight_queue.put_nowait(weights)
            except:
                pass

    # 为Collector提供Actor参数
    def get_actor_state_dict(self, device=torch.device("cpu")):
        state_dict = self.actor.model.state_dict()
        state_dict_on_target_device = {
            key: tensor.to(device) for key, tensor in state_dict.items()
        }
        return state_dict_on_target_device

     # 训练1次
    def train(self):
        try:
            # 阻塞式等待, 直到有批次可用
            state_batch, action_batch, _, next_state_batch, reward_batch, terminated_batch, _, index_batch, weight_batch = self.sample_queue.get(timeout=10)
        except queue.Empty:
            print(f"[{self.index}] 等待采样批次超时...")
            return
        self.step += 1
        if index_batch is not None:
            weight_batch = torch.from_numpy(weight_batch).float().unsqueeze(1).to(self.device)
        else:
            weight_batch = torch.ones((self.batch_shape, 1)).float().to(self.device)
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        action_batch = torch.from_numpy(action_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        terminated_batch = torch.from_numpy(terminated_batch).float().to(self.device)

        with torch.no_grad():
            next_action_batch, _ = self.actor_target.get_action(next_state_batch, deterministic=True)
            eval_noise = (torch.randn_like(next_action_batch) * self.eval_noise_std).clamp(-self.eval_noise_bound, self.eval_noise_bound)
            self.eval_noise_std = max(self.eval_noise_std * self.eval_noise_decay, 0.05)
            self.eval_noise_bound = max(self.eval_noise_bound * self.eval_noise_decay, 0.01)
            next_action_batch = (next_action_batch + eval_noise).clamp(-self.action_bound, self.action_bound)
            next_q_batch_1, next_q_batch_2 = self.critic_target.get_value(next_state_batch, next_action_batch)
            next_q_batch = torch.min(next_q_batch_1, next_q_batch_2)
            target_q_batch = reward_batch + self.reward_gamma * (1 - terminated_batch) * next_q_batch

        self.critic.loss, td_error_batch = self.critic.train(state_batch, action_batch, target_q_batch, weight_batch)
        if self.step % self.actor_train_freq == 0:
            self.actor.loss = self.actor.train(state_batch)
            self._distribute_weights()
        if self.step % self.update_freq == 0:
            update_target_model(self.actor.model, self.actor_target.model, self.update_tau)
            update_target_model(self.critic.model, self.critic_target.model, self.update_tau)
        if index_batch is not None:
            td_error_batch = td_error_batch.squeeze(1).detach().cpu().numpy()
            self.error_queue.put((index_batch, td_error_batch))

    def model_save(self, file_path):
        torch.save(self.critic_target.state_dict(), f"{file_path}/Agent_{self.index}_Critic_model.h5")
        torch.save(self.actor_target.state_dict(), f"{file_path}/Agent_{self.index}_Actor_model.h5")

    def model_load(self, file_path, index=None):
        if index is None: index = self.index
        critic_path = f"{file_path}/Agent_{index}_Critic_model.h5"
        actor_path = f"{file_path}/Agent_{index}_Actor_model.h5"
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.critic_target.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.actor_target.load_state_dict(torch.load(actor_path, map_location=self.device))


class TD3_Collector():
    def __init__(self, *args, **kwargs):
        # --- 1. 弹出所有非简单类型的对象 ---
        self.experience_queue = kwargs.pop("experience_queue")
        self.weight_queue = kwargs.pop("weight_queue")

        self.config = copy.deepcopy(kwargs)

        # --- 2. 获取核心配置 ---
        self.index = self.config["index"]
        self.state_shape = self.config["state_shape"]
        self.action_shape = self.config["action_shape"]
        self.device = torch.device(self.config.get("device", "cpu"))

        # --- 3. 获取算法超参数 ---
        self.actor_hidden_shape = self.config.get("actor_hidden_shape", [256, 256])
        self.actor_activation = self.config.get("actor_activation", "tanh")
        self.hidden_activation = self.config.get("hidden_activation", "relu")
        self.action_bound = self.config.get("action_bound", 1.0)
        self.step = 0
        self.check_weights_freq = self.config.get("check_weights_freq", 10)

        # --- 4. 实例化探索Actor (在 CPU 上) ---
        actor_kwargs = {
            "state_shape": self.state_shape, "action_shape": self.action_shape,
            "hidden_shape": self.actor_hidden_shape, "activation": self.actor_activation,
            "hidden_activation": self.hidden_activation,
        }
        self.actor = Deterministic_Actor(**actor_kwargs).to(self.device)
        self.actor.model.eval()

        # --- 5. 添加批处理 ---
        self.batch_send_freq = self.config.get("batch_send_freq", 100)
        self.local_batch = []

    # 从Learner接收当前Actor参数
    def _check_for_new_weights(self):
        try:
            new_weights = self.weight_queue.get_nowait()
            self.set_actor_state_dict(new_weights)
        except queue.Empty:
            pass
        except:
            print(f"Collector {self.index} 权重同步失败.")

    # 为Learner更新Actor参数
    def set_actor_state_dict(self, actor_weights_state_dict):
        self.actor.load_state_dict(actor_weights_state_dict)

    def get_action(self, state):
        self.step += 1
        if self.step % self.check_weights_freq == 0:
            self._check_for_new_weights()
        if len(state.shape) == 1:
            state_batch = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        else:
            state_batch = torch.from_numpy(state).float().to(self.device)
        action_batch, _ = self.actor.get_action(state_batch, deterministic=True)
        action_batch = action_batch.detach().cpu().numpy()
        if len(state.shape) == 1:
            return action_batch[0], None
        else:
            return action_batch, None

    # 将交互数据存入Learner Buffer
    def remember(self, experience):
        self.local_batch.append(experience)
        if len(self.local_batch) >= self.batch_send_freq:
            try:
                self.experience_queue.put_nowait(self.local_batch.copy())
            except queue.Full:
                print(f"警告: Collector {self.index} 数据队列已满, 丢弃数据.")
            self.local_batch.clear()