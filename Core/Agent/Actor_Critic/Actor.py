import copy
import torch
from torch.distributions import Normal
from Core.Agent.Actor_Critic.Base_MLP import Base_MLP


class Deterministic_Actor():
    def __init__(self, *args, **kwargs):
        self.config = copy.deepcopy(kwargs)
        self.mlp_config = {
            "input_shape": self.config["state_shape"],
            "output_shape": self.config["action_shape"],
            "hidden_shape": self.config["hidden_shape"],
            "activation": self.config.get("activation", "tanh"),
            "hidden_activation": self.config.get("hidden_activation", "relu")
        }
        self.model = Base_MLP(**self.mlp_config)

    def get_action(self, *state_batch, deterministic=False, with_log_prob=False):
        action_batch = self.model(*state_batch)
        log_prob_batch = None
        return action_batch, log_prob_batch

    def to(self, device):
        self.model.to(device)
        return self

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class Gaussian_Actor():
    def __init__(self, *args, **kwargs):
        self.config = copy.deepcopy(kwargs)
        self.log_prob_epsilon = self.config.get("log_prob_epsilon", 1e-6)
        self.min_log_std = self.config.get("min_log_std", -20.0)
        self.max_log_std = self.config.get("max_log_std", 2.0)
        self.mlp_config = {
            "input_shape": self.config["state_shape"],
            "output_shape": (2, self.config["action_shape"]),
            "hidden_shape": self.config["hidden_shape"],
            "activation": self.config.get("activation", ["tanh", "linear"]),
            "hidden_activation": self.config.get("hidden_activation", "relu")
        }
        self.model = Base_MLP(**self.mlp_config)

    def get_action(self, *state_batch, deterministic=False, with_log_prob=False):
        mu_batch, log_std_batch = self.model(*state_batch)
        log_std_batch = torch.clamp(log_std_batch, self.min_log_std, self.max_log_std)
        std_batch = torch.exp(log_std_batch)
        dist_batch = Normal(mu_batch, std_batch)
        u_batch = dist_batch.rsample()
        action_batch = torch.tanh(u_batch)
        if deterministic:
            action_batch = torch.tanh(mu_batch)
            log_prob_batch = None
        else:
            if with_log_prob:
                # log_prob(action) = log_prob(u) - log(1 - action^2 + eps)
                log_prob_u_batch = dist_batch.log_prob(u_batch)
                correction = torch.log(1.0 - action_batch.pow(2) + self.log_prob_epsilon)
                log_prob_batch = (log_prob_u_batch - correction).sum(dim=-1, keepdim=True)
            else:
                log_prob_batch = None
        return action_batch, log_prob_batch

    def to(self, device):
        self.model.to(device)
        return self

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)