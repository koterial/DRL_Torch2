import torch
import torch.nn as nn
import numpy as np


def batch_norm(batch, min_std=1e-5):
    batch = (batch - np.mean(batch)) / (np.std(batch) + min_std)
    return batch


def clip_by_local_norm(parameters, norm=0.5):
    nn.utils.clip_grad_norm_(parameters=parameters, max_norm=norm)


def update_target_model(model: torch.nn.Module, target_model: torch.nn.Module, update_tau):
    for model_parameters, target_model_parameters in zip(model.parameters(), target_model.parameters()):
        target_model_parameters.data.copy_(update_tau * model_parameters.data + (1 - update_tau) * target_model_parameters.data)


def discount_reward(reward_batch, terminated_batch, reward_gamma):
    discount_reward_batch = []
    discount_reward = 0
    for reward, terminated in zip(reward_batch[::-1], terminated_batch[::-1]):
        if terminated:
            discount_reward = 0
        discount_reward = reward + reward_gamma * discount_reward * (1 - terminated)
        discount_reward_batch.insert(0, discount_reward)
    discount_reward_batch = np.stack(discount_reward_batch)
    return discount_reward_batch


def gae(td_error_batch, terminated_batch, reward_gamma, lamba):
    gae_batch = []
    gae = 0
    for td_error, terminated in zip(td_error_batch[::-1], terminated_batch[::-1]):
        if terminated:
            gae = 0
        gae = td_error + reward_gamma * lamba * gae * (1 - terminated)
        gae_batch.insert(0, gae)
    gae_batch = np.stack(gae_batch)
    return gae_batch