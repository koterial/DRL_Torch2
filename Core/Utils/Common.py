import os
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from easydict import EasyDict


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("设置实验seed:", seed)


def save_gif(frame_list, file_path, fps=30):
    if os.path.isfile(file_path):
        os.remove(file_path)
    patch = plt.imshow(frame_list[0])
    plt.axis("off")
    def animate(i):
        patch.set_data(frame_list[i])
    anim = animation.FuncAnimation(fig=plt.gcf(), func=animate, frames=len(frame_list), interval=1)
    anim.save(file_path, writer="pillow", fps=fps)


def config_load(file_path):
    with open(file_path, "r") as file:
        config_data = json.load(file)
    return EasyDict(config_data)


def config_save(config_data, file_path):
    with open(file_path + "/config.json", "w") as file:
        json.dump(EasyDict(config_data), file)