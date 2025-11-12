import copy
import numpy as np


class Gaussian_Noise():
    def __init__(self, *args, **kwargs):
        self.config = copy.deepcopy(kwargs)
        self.class_name = "Gaussian"

        self.index = self.config["index"]
        self.action_shape = self.config["action_shape"]
        self.action_dim = sum(self.action_shape)
        self.mu = self.config.get("mu", 0.0)
        self.std = self.config.get("std", 0.4)
        self.scale = self.config.get("scale", 0.1)
        self.bound = self.config.get("bound", 0.2)
        self.decay = self.config.get("decay", 0.999)
        self.reset()

    def reset(self):
        self.state = self.mu * np.ones(shape=self.action_dim)

    def get_noise(self):
        self.state = np.random.normal(self.mu, self.std, size=self.action_dim)
        return np.clip(self.state * self.scale, -1 * self.bound, self.bound)

    def bound_decay(self):
        self.scale = max(self.scale * self.decay, 0.01)
        self.bound = max(self.bound * self.decay, 0.01)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    noise_config = {
        "index": 1,
        "action_shape": [1],
        "std": 0.4,
        "scale": 0.1,
        "bound": 0.2
    }
    noise = Gaussian_Noise(**noise_config)
    noise_list = []
    for _ in range(100000):
        noise_list.append(noise.get_noise())

    plt.plot(noise_list)
    plt.show()