import os
import torch
import random
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from Utils.Common import clip_by_local_norm
from Replay_Buffer.Replay_Buffer import Replay_Buffer, Prioritized_Replay_Buffer
from DDPG.DDPG import DDPG_Critic, DDPG_Actor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3_Agent():
    def __init__(self, agent_index, state_shape, action_shape, critic_unit_num, critic_layer_num, critic_lr,
                 actor_unit_num, actor_layer_num, actor_lr,
                 batch_size, buffer_size, gamma=0.95, tau=0.01, update_freq=1, activation="linear", prioritized_replay=False,
                 alpha=0.6, beta=0.4, beta_increase=1e-3, max_priority=1, min_priority=0.01, clip_norm=0.5
                 ):
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.critic_unit_num = critic_unit_num
        self.critic_layer_num = critic_layer_num
        self.critic_lr = critic_lr
        self.actor_unit_num = actor_unit_num
        self.actor_layer_num = actor_layer_num
        self.activation = activation
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.tau = tau
        self.update_freq = update_freq
        self.update_counter = 0
        self.clip_norm = clip_norm

        self.train_critic_1 = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                        action_shape=self.action_shape,
                                        unit_num=self.critic_unit_num, layer_num=self.critic_layer_num,
                                        lr=self.critic_lr,
                                        clip_norm=self.clip_norm)
        self.target_critic_1 = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                        action_shape=self.action_shape,
                                        unit_num=self.critic_unit_num, layer_num=self.critic_layer_num,
                                        lr=self.critic_lr,
                                        clip_norm=self.clip_norm)
        self.train_critic_2 = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                          action_shape=self.action_shape,
                                          unit_num=self.critic_unit_num, layer_num=self.critic_layer_num,
                                          lr=self.critic_lr,
                                          clip_norm=self.clip_norm)
        self.target_critic_2 = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                           action_shape=self.action_shape,
                                           unit_num=self.critic_unit_num, layer_num=self.critic_layer_num,
                                           lr=self.critic_lr,
                                           clip_norm=self.clip_norm)
        self.target_actor = DDPG_Actor(agent_index=self.agent_index, state_shape=self.state_shape,
                                       action_shape=self.action_shape,
                                       unit_num=self.actor_unit_num, layer_num=self.actor_layer_num,
                                       lr=self.actor_lr,
                                       critic=self.train_critic_1, activation=self.activation, clip_norm=self.clip_norm)
        self.train_actor = DDPG_Actor(agent_index=self.agent_index, state_shape=self.state_shape,
                                       action_shape=self.action_shape,
                                       unit_num=self.actor_unit_num, layer_num=self.actor_layer_num,
                                       lr=self.actor_lr,
                                       critic=self.train_critic_1, activation=self.activation, clip_norm=self.clip_norm)
        self.target_critic_1.model.load_state_dict(self.train_critic_1.model.state_dict())
        self.target_critic_2.model.load_state_dict(self.train_critic_2.model.state_dict())
        self.target_actor.model.load_state_dict(self.train_actor.model.state_dict())

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increase = beta_increase
        self.prioritized_replay = prioritized_replay
        self.max_priority = max_priority
        self.min_priority = min_priority
        if self.prioritized_replay:
            self.replay_buffer = Prioritized_Replay_Buffer(buffer_size, self.alpha, self.beta, self.beta_increase,
                                                           self.max_priority, self.min_priority)
        else:
            self.replay_buffer = Replay_Buffer(buffer_size)

    def get_action(self, state):
        action = self.train_actor.get_action(torch.asarray([state], dtype=torch.float32).to(device))[0]
        return action.cpu().detach().numpy()

    def get_target_action(self, state):
        action = self.target_actor.get_action(torch.asarray([state], dtype=torch.float32).to(device))[0]
        return action.cpu().detach().numpy()

    def train(self):
        self.update_counter += 1
        if self.prioritized_replay:
            state_batch, action_batch, next_state_batch, reward_batch, done_batch, index_batch, weight_batch = self.replay_buffer.sample(
                self.batch_size)
        else:
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.replay_buffer.sample(
                self.batch_size)
            weight_batch = torch.ones(size=(self.batch_size,), dtype=torch.float32)
        next_action_batch = self.target_actor.get_action(next_state_batch)
        next_q_batch = self.target_critic.model([next_state_batch]+[next_action_batch])
        target_q_batch = reward_batch[:, None] + self.gamma * next_q_batch * (1 - done_batch[:, None].astype(int))
        td_error_batch = self.train_critic.train(state_batch, action_batch, target_q_batch, weight_batch)
        if self.prioritized_replay:
            self.replay_buffer.batch_update(index_batch, np.sum(td_error_batch, axis=1))
        if self.update_counter % self.update_freq == 0:
            self.train_actor.train(state_batch)
            self.model_update(self.tau)

    def model_update(self, tau):
        def update_target_model(model:torch.nn.Module, target_model:torch.nn.Module):
            model_weight = model.load_state_dict()
            target_model_weight = target_model.load_state_dict()
            new_model_weight = tau * model_weight + (1-tau) * target_model_weight
            target_model.load_state_dict(new_model_weight)
        update_target_model(self.train_critic_1.model, self.target_critic_1.model)
        update_target_model(self.train_critic_2.model, self.target_critic_2.model)
        update_target_model(self.train_actor.model, self.target_actor.model)


    def model_save(self, file_path, seed):
        if os.path.exists(file_path):
            pass
        else:
            os.makedirs(file_path)
        torch.save(self.target_critic_1.model.state_dict(), "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
        torch.save(self.target_critic_2.model.state_dict(), "/Agent_{}_Critic_2_model.h5".format(self.agent_index))
        torch.save(self.target_actor.model.state_dict(), "/Agent_{}_Actor_model.h5".format(self.agent_index))
        file = open(file_path + "/Agent_{}_train.log".format(self.agent_index), "w")
        file.write(
            "type:" + str("DDPG") +
            "\nseed:" + str(seed) +
            "\nstate_shape:" + str(self.state_shape) +
            "\naction_shape:" + str(self.action_shape) +
            "\ncritic_unit_num:" + str(self.critic_unit_num) +
            "\ncritic_layers_num:" + str(self.critic_layer_num) +
            "\ncritic_lr:" + str(self.critic_lr) +
            "\nactor_unit_num:" + str(self.actor_unit_num) +
            "\nactor_layers_num:" + str(self.actor_layer_num) +
            "\nactivation:" + str(self.activation) +
            "\nactor_lr:" + str(self.actor_lr) +
            "\ngamme:" + str(self.gamma) +
            "\ntau:" + str(self.tau) +
            "\nupdate_freq:" + str(self.update_freq) +
            "\nbatch_size:" + str(self.batch_size) +
            "\nbuffer_size:" + str(self.buffer_size) +
            "\nPER:" + str(self.prioritized_replay) +
            "\nalpha:" + str(self.alpha) +
            "\nbeta:" + str(self.beta) +
            "\nbeta_increase:" + str(self.beta_increase) +
            "\nmax_priority:" + str(self.max_priority) +
            "\nmin_priority:" + str(self.min_priority) +
            "\nclip_norm:" + str(self.clip_norm)
        )

    def model_load(self, file_path, agent_index=None):
        if agent_index:
            self.target_critic_1.model.load_state_dict(file_path + "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
            self.train_critic_1.model.load_state_dict(file_path + "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
            self.target_critic_2.model.load_state_dict(file_path + "/Agent_{}_Critic_2_model.h5".format(self.agent_index))
            self.train_critic_2.model.load_state_dict(file_path + "/Agent_{}_Critic_2_model.h5".format(self.agent_index))
            self.target_actor.model.load_state_dict(file_path + "/Agent_{}_Actor_model.h5".format(self.agent_index))
            self.train_actor.model.load_state_dict(file_path + "/Agent_{}_Actor_model.h5".format(self.agent_index))
        else:
            self.target_critic_1.model.load_state_dict(file_path + "/Agent_{}_Critic_1_model.h5".format(agent_index))
            self.train_critic_1.model.load_state_dict(file_path + "/Agent_{}_Critic_1_model.h5".format(agent_index))
            self.target_critic_2.model.load_state_dict(file_path + "/Agent_{}_Critic_2_model.h5".format(agent_index))
            self.train_critic_2.model.load_state_dict(file_path + "/Agent_{}_Critic_2_model.h5".format(agent_index))
            self.target_actor.model.load_state_dict(file_path + "/Agent_{}_Actor_model.h5".format(agent_index))
            self.train_actor.model.load_state_dict(file_path + "/Agent_{}_Actor_model.h5".format(agent_index))