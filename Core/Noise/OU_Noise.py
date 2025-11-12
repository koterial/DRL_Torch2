import copy
import numpy as np


class OU_Noise():
    def __init__(self, *args, **kwargs):
        self.config = copy.deepcopy(kwargs)
        self.class_name = "OU"

        self.index = self.config["index"]
        self.action_shape = self.config["action_shape"]
        self.action_dim = sum(self.action_shape)
        self.mu = self.config.get("mu", 0.0)
        self.theta = self.config.get("theta", 0.15)
        self.std = self.config.get("std", 0.25)
        self.dt = self.config.get("dt", 1e-2)
        self.scale = self.config.get("scale", 0.1)
        self.bound = self.config.get("bound", 0.2)
        self.decay = self.config.get("decay", 0.999)
        self.reset()

    def reset(self):
        self.state = self.mu * np.ones(shape=self.action_dim)

    def get_noise(self):
        x = self.state
        # 第一部分为均值回归过程, 第二部分为布朗运动随机项
        dx = self.theta * (self.mu - x) * self.dt + self.std * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state = x + dx
        return np.clip(self.state * self.scale, -1 * self.bound, self.bound)

    def bound_decay(self):
        self.scale = max(self.scale * self.decay, 0.01)
        self.bound = max(self.bound * self.decay, 0.01)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    noise_config = {
        "index": 1,
        "action_shape": [1]
    }
    noise = OU_Noise(**noise_config)
    noise_list = []
    for _ in range(100000):
        noise_list.append(noise.get_noise())

    plt.plot(noise_list)
    plt.show()