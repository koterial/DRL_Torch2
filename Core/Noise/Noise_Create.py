import copy
from Core.Noise.Gaussian_Noise import Gaussian_Noise
from Core.Noise.OU_Noise import OU_Noise


def noise_create(explore_noise_config):
    init_kwargs = copy.deepcopy(explore_noise_config)
    noise_class_name = init_kwargs.pop("class")

    if noise_class_name == "Gaussian":
        noise = Gaussian_Noise(**init_kwargs)
    elif noise_class_name == "OU":
        noise = OU_Noise(**init_kwargs)
    else:
        raise ValueError(f"未知的 Noise 类别: {noise_class_name}")

    return noise