import os
import torch
import random
import numpy as np
import torch.nn as nn

def clip_by_local_norm(parameters, norm):
    nn.utils.clip_grad_norm(parameters=parameters, max_norm=norm)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)