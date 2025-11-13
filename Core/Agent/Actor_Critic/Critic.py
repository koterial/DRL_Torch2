import copy
import torch
from Core.Agent.Actor_Critic.Base_MLP import Base_MLP


class V_Critic():
    def __init__(self, *args, **kwargs):
        self.config = copy.deepcopy(kwargs)
        self.mlp_config = {
            "input_shape": self.config["state_shape"],
            "output_shape": 1,
            "hidden_shape": self.config["hidden_shape"],
            "activation": self.config.get("activation", "linear"),
            "hidden_activation": self.config.get("hidden_activation", "relu")
        }
        self.model = Base_MLP(**self.mlp_config)

    def get_value(self, *state_batch):
        return self.model(*state_batch)

    def to(self, device):
        self.model.to(device)
        return self

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class Q_Critic():
    def __init__(self, *args, **kwargs):
        self.config = copy.deepcopy(kwargs)
        self.mlp_config = {
            "input_shape": [self.config["state_shape"], self.config["action_shape"]],
            "output_shape": 1,
            "hidden_shape": self.config["hidden_shape"],
            "activation": self.config.get("activation", "linear"),
            "hidden_activation": self.config.get("hidden_activation", "relu")
        }
        self.model = Base_MLP(**self.mlp_config)

    def get_value(self, *state_action_batch):
        return self.model(*state_action_batch)

    def to(self, device):
        self.model.to(device)
        return self

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class Twin_Q_Critic():
    def __init__(self, *args, **kwargs):
        self.config = copy.deepcopy(kwargs)
        self.mlp_config = {
            "input_shape": [self.config["state_shape"], self.config["action_shape"]],
            "output_shape": (2, 1),
            "hidden_shape": self.config["hidden_shape"],
            "activation": self.config.get("activation", ["linear", "linear"]),
            "hidden_activation": self.config.get("hidden_activation", "relu")
        }
        self.model = Base_MLP(**self.mlp_config)

    def get_value(self, *state_action_batch):
        return self.model(*state_action_batch)

    def to(self, device):
        self.model.to(device)
        return self

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)