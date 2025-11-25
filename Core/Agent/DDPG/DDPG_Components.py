import copy
import torch
import torch.nn.functional as F
from Core.Agent.Actor_Critic.Actor import Deterministic_Actor
from Core.Agent.Actor_Critic.Critic import Q_Critic
from Core.Agent.Utils.Common import clip_by_local_norm, update_target_model
from Core.Buffer.Buffer_Create import buffer_create

torch.set_default_dtype(torch.float32)


class DDPG_Agent():
    def __init__(self, *args, **kwargs):
        #  --- 1. 弹出所有非简单类型的对象 ---
        self.buffer_config = kwargs.pop("buffer_config")

        self.config = copy.deepcopy(kwargs)

        # --- 2. 获取核心配置 ---
        self.index = self.config.get("index", 0)
        self.state_shape = self.config["state_shape"]
        self.action_shape = self.config["action_shape"]
        self.device = torch.device(self.config.get("device", "cuda"))

        # --- 3. 获取算法超参数 ---
        self.actor_lr = self.config.get("actor_lr", 1e-3)
        self.critic_lr = self.config.get("critic_lr", 1e-3)
        self.reward_gamma = self.config.get("reward_gamma", 0.99)
        self.update_tau = self.config.get("update_tau", 0.005)
        self.clip_norm = self.config.get("clip_norm", 0.5)
        self.batch_shape = int(self.config.get("batch_shape", 256))

        self.actor_hidden_shape = self.config.get("actor_hidden_shape", [256, 256])
        self.critic_hidden_shape = self.config.get("critic_hidden_shape", [256, 256])
        self.actor_activation = self.config.get("actor_activation", "tanh")
        self.critic_activation = self.config.get("critic_activation", "linear")
        self.hidden_activation = self.config.get("hidden_activation", "relu")

        self.action_bound = self.config.get("action_bound", 1.0)
        self.learn_step = 0
        self.collect_step = 0

        # --- 4. 实例化训练Critic (self.device) ---
        critic_kwargs = {
            "state_shape": self.state_shape, "action_shape": self.action_shape,
            "hidden_shape": self.critic_hidden_shape, "activation": self.critic_activation,
            "hidden_activation": self.hidden_activation, "lr": self.critic_lr,
            "clip_norm": self.clip_norm, "device": self.device
        }
        self.critic = DDPG_Critic(**critic_kwargs).to(self.device)
        self.critic_target = DDPG_Critic(**critic_kwargs).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # --- 5. 实例化训练Actor (self.device) ---
        actor_kwargs = {
            "state_shape": self.state_shape, "action_shape": self.action_shape,
            "hidden_shape": self.actor_hidden_shape, "activation": self.actor_activation,
            "hidden_activation": self.hidden_activation, "lr": self.actor_lr,
            "clip_norm": self.clip_norm, "critic": self.critic, "device": self.device
        }
        self.actor = DDPG_Actor(**actor_kwargs).to(self.device)
        self.actor_target = DDPG_Actor(**actor_kwargs).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        #  --- 6. 实例化经验回放池 ---
        self.buffer_config["state_shape"] = self.state_shape
        self.buffer_config["action_shape"] = self.action_shape
        self.replay_buffer = buffer_create(self.buffer_config)
        self.prioritized_replay = True if self.buffer_config.get("class") == "Prioritized_Replay_Buffer" else False

    def get_action(self, state, deterministic=False):
        self.collect_step += 1
        if len(state.shape) == 1:
            state_batch = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        else:
            state_batch = torch.from_numpy(state).float().to(self.device)
        action_batch, _ = self.actor.get_action(state_batch, deterministic)
        action_batch = action_batch.detach().cpu().numpy()
        if len(state.shape) == 1:
            return action_batch[0], None
        else:
            return action_batch, None

    def remember(self, experience):
        self.replay_buffer.remember([experience])

    def train(self):
        if self.replay_buffer.size() < self.batch_shape:
            return
        self.learn_step += 1
        state_batch, action_batch, _, next_state_batch, reward_batch, terminated_batch, _, index_batch, weight_batch = self.replay_buffer.sample(
            self.batch_shape)
        if self.prioritized_replay:
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
            next_q_batch = self.critic_target.get_value(next_state_batch, next_action_batch)
            target_q_batch = reward_batch + self.reward_gamma * (1 - terminated_batch) * next_q_batch

        self.critic.loss, td_error_batch = self.critic.train(state_batch, action_batch, target_q_batch, weight_batch)
        self.actor.loss = self.actor.train(state_batch)
        update_target_model(self.actor.model, self.actor_target.model, self.update_tau)
        update_target_model(self.critic.model, self.critic_target.model, self.update_tau)

        if self.prioritized_replay:
            self.replay_buffer.batch_update(index_batch, td_error_batch.squeeze(1).detach().cpu().numpy())

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


class DDPG_Actor(Deterministic_Actor):
    def __init__(self, *args, **kwargs):
        self.critic = kwargs.pop("critic")
        super().__init__(*args, **kwargs)
        self.lr = self.config["lr"]
        self.clip_norm = self.config["clip_norm"]
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = 0

    def train(self, state_batch):
        new_action_batch, _ = self.get_action(state_batch, deterministic=False)
        q_batch = self.critic.get_value(state_batch, new_action_batch)
        loss = -1 * torch.mean(q_batch)
        self.opt.zero_grad()
        loss.backward()
        clip_by_local_norm(self.model.parameters(), self.clip_norm)
        self.opt.step()
        return loss.item()


class DDPG_Critic(Q_Critic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = self.config["lr"]
        self.clip_norm = self.config["clip_norm"]
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = 0

    def train(self, state_batch, action_batch, target_q_batch, weight_batch):
        q_batch = self.get_value(state_batch, action_batch)
        td_error_batch = target_q_batch - q_batch
        loss = F.mse_loss(q_batch, target_q_batch.detach(), reduction='none')
        loss = torch.mean(weight_batch * loss)
        self.opt.zero_grad()
        loss.backward()
        clip_by_local_norm(self.model.parameters(), self.clip_norm)
        self.opt.step()
        return loss.item(), torch.abs(td_error_batch)