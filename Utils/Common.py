import torch
import torch.nn as nn

def clip_by_local_norm(parameters, norm):
    nn.utils.clip_grad_norm(parameters=parameters, max_norm=norm)