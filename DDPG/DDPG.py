import os
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from Utils.Common import clip_by_local_norm
from Replay_Buffer.Replay_Buffer import Replay_Buffer, Prioritized_Replay_Buffer

class DDPG_Agent():
    def __init__(self, agent_index, state_shape, action_shape, critic_units_num, critic_layer_num, critic_lr,
                 actor_units_num, actor_layer_num, actor_lr,
                 batch_size, buffer_size, gamma, tau, update_freq=1, activation="linear", prioritized_replay=False,
                 alpha=0.6, beta=0.4, beta_increase=1e-3, max_priority=1, min_priority=0.01, clip_norm=0.5
                 ):
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.critic_units_num = critic_units_num
        self.critic_layer_num = critic_layer_num
        self.critic_lr = critic_lr
        self.actor_units_num = actor_units_num
        self.actor_layer_num = actor_layer_num
        self.activation = activation
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.tau = tau
        self.update_freq = update_freq
        self.update_counter = 0
        self.clip_norm = clip_norm

        self.train_critic = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                        action_shape=self.action_shape,
                                        units_num=self.critic_units_num, layer_num=self.critic_layer_num,
                                        lr=self.critic_lr,
                                        clip_norm=self.clip_norm)
        self.target_critic = DDPG_Critic(agent_index=self.agent_index, state_shape=self.state_shape,
                                        action_shape=self.action_shape,
                                        units_num=self.critic_units_num, layer_num=self.critic_layer_num,
                                        lr=self.critic_lr,
                                        clip_norm=self.clip_norm)
        self.target_actor = DDPG_Actor(agent_index=self.agent_index, state_shape=self.state_shape,
                                       action_shape=self.action_shape,
                                       units_num=self.actor_units_num, layer_num=self.actor_layer_num,
                                       lr=self.actor_lr,
                                       critic=self.train_critic, activation=self.activation, clip_norm=self.clip_norm)
        self.train_actor = DDPG_Actor(agent_index=self.agent_index, state_shape=self.state_shape,
                                       action_shape=self.action_shape,
                                       units_num=self.actor_units_num, layer_num=self.actor_layer_num,
                                       lr=self.actor_lr,
                                       critic=self.train_critic, activation=self.activation, clip_norm=self.clip_norm)
        self.target_critic.model.load_state_dict(self.train_critic.model.state_dict())
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
        action = self.train_actor.get_action(torch.asarray([state]))[0]
        return action.item()

    def get_target_action(self, state):
        action = self.target_actor.get_action(torch.asarray([state]))[0]
        return action.item()

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
        update_target_model(self.train_critic.model, self.target_critic.model)
        update_target_model(self.train_actor.model, self.target_actor.model)


    def model_save(self):
        pass

    def model_load(self):
        pass

class DDPG_Critic():
    def __init__(self, agent_index, state_shape, action_shape, units_num, layer_num, lr, activation="linear", clip_norm=0.5):
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.units_num = units_num
        self.layer_num = layer_num
        self.activation = activation
        self.lr = lr
        self.clip_norm = clip_norm
        self.model = DDPG_Critic_Model(self.agent_index, self.state_shape, self.action_shape, self.units_num, self.layer_num, self.activation)
        self.opt = torch.optim.Adam(self.model.parameters(), self.lr)

    def train(self, state_batch, action_batch, target_q_batch, weight_batch):
        q_batch = self.model([state_batch, action_batch])
        td_error_batch = torch.square(target_q_batch - q_batch)
        error = torch.mean(
            torch.tensor([torch.sum(td_error * weight) for td_error, weight in zip(td_error_batch, weight_batch)]))
        self.opt.zero_grad()
        error.backward()
        clip_by_local_norm(self.model.parameters())
        self.opt.step()
        return td_error_batch

class DDPG_Actor():
    def __init__(self, agent_index, state_shape, action_shape, units_num, layer_num, lr, critic, activation="linear", clip_norm=0.5):
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.units_num = units_num
        self.layer_num = layer_num
        self.activation = activation
        self.critic = critic
        self.lr = lr
        self.clip_norm = clip_norm
        self.model = DDPG_Actor_Model(self.agent_index, self.state_shape, self.action_shape, self.units_num, self.layer_num, self.activation)
        self.opt = torch.optim.Adam(self.model.parameters(), self.lr)

    def get_action(self, state_batch):
        action_batch = self.model(state_batch)
        return action_batch

    def train(self, state_batch):
        new_action_batch = self.model(state_batch)
        action_batch = new_action_batch.clone()
        q_batch = self.critic.model([state_batch] + [action_batch])
        policy_regularization = torch.mean(torch.square(new_action_batch))
        loss = -torch.mean(q_batch) + 1e-3 * policy_regularization
        self.opt.zero_grad()
        loss.backward()
        clip_by_local_norm(self.model.parameters())
        self.opt.step()

class DDPG_Critic_Model(nn.Module):
    def __init__(self, agent_index, state_shape, action_shape, units_num, layer_num, activation="linear"):
        super(DDPG_Critic_Model, self).__init__()
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.input_shape = sum(self.state_shape) + sum(self.action_shape)
        self.output_shape = 1
        self.units_num = units_num
        self.layer_num = layer_num
        self.activation = activation
        self.hidden_layer_list = nn.ModuleList()
        for each in range(self.layer_num):
            if each == 0:
                self.hidden_layer_list.append(nn.Sequential(
                    OrderedDict([
                        ("Agent_{}_critic_hidden_{}".format(self.agent_index, each), nn.Linear(self.input_shape, self.units_num)),
                        ("Agent_{}_critic_hidden_{}_activation".format(self.agent_index, each), nn.ReLU())
                    ])))
            else:
                self.hidden_layer_list.append(nn.Sequential(
                    OrderedDict([
                        ("Agent_{}_critic_hidden_{}".format(self.agent_index, each), nn.Linear(self.units_num, self.units_num)),
                        ("Agent_{}_critic_hidden_{}_activation".format(self.agent_index, each), nn.ReLU())
                    ])))
        if self.activation == "sigmoid":
            self.output_layer = nn.Sequential(
                OrderedDict([
                    ("Agent_{}_critic_output".format(self.agent_index), nn.Linear(self.units_num, self.output_shape)),
                    ("Agent_{}_critic_output_activation".format(self.agent_index), nn.Sigmoid())
                ]))
        elif self.activation == "tanh":
            self.output_layer = nn.Sequential(
                OrderedDict([
                    ("Agent_{}_critic_output".format(self.agent_index), nn.Linear(self.units_num, self.output_shape)),
                    ("Agent_{}_critic_output_activation".format(self.agent_index), nn.Tanh())
                ]))
        else:
            self.output_layer = nn.Sequential(
                OrderedDict([
                    ("Agent_{}_critic_output".format(self.agent_index), nn.Linear(self.units_num, self.output_shape))
                ]))

    def forward(self, input_list):
        x = torch.concatenate(input_list)
        for hidden_layer in self.hidden_layer_list:
            x = hidden_layer(x)
        output = self.output_layer(x)
        return output

class DDPG_Actor_Model(nn.Module):
    def __init__(self, agent_index, state_shape, action_shape, units_num, layer_num, activation="linear"):
        super(DDPG_Actor_Model, self).__init__()
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.input_shape = sum(self.state_shape)
        self.output_shape = self.action_shape
        self.units_num = units_num
        self.layer_num = layer_num
        self.activation = activation
        self.hidden_layer_list = nn.ModuleList()
        for each in range(self.layer_num):
            if each == 0:
                self.hidden_layer_list.append(nn.Sequential(
                    OrderedDict([
                        ("Agent_{}_actor_hidden_{}".format(self.agent_index, each), nn.Linear(self.input_shape, self.units_num)),
                        ("Agent_{}_actor_hidden_{}_activation".format(self.agent_index, each), nn.ReLU())
                    ])))
            else:
                self.hidden_layer_list.append(nn.Sequential(
                    OrderedDict([
                        ("Agent_{}_actor_hidden_{}".format(self.agent_index, each), nn.Linear(self.units_num, self.units_num)),
                        ("Agent_{}_actor_hidden_{}_activation".format(self.agent_index, each), nn.ReLU())
                    ])))
        self.output_layer_list = nn.ModuleList()
        for each, shape in enumerate(self.output_shape):
            if self.activation == "sigmoid":
                self.output_layer_list.append(nn.Sequential(
                    OrderedDict([
                        ("Agent_{}_actor_output_{}".format(self.agent_index, each), nn.Linear(self.units_num, shape)),
                        ("Agent_{}_actor_output_{}_activation".format(self.agent_index, each), nn.Sigmoid())
                    ])))
            elif self.activation == "tanh":
                self.output_layer_list.append(nn.Sequential(
                    OrderedDict([
                        ("Agent_{}_actor_output_{}".format(self.agent_index, each), nn.Linear(self.units_num, shape)),
                        ("Agent_{}_actor_output_{}_activation".format(self.agent_index, each), nn.Tanh())
                    ])))
            elif self.activation == "softmax":
                self.output_layer_list.append(nn.Sequential(
                    OrderedDict([
                        ("Agent_{}_actor_output_{}".format(self.agent_index, each), nn.Linear(self.units_num, shape)),
                        ("Agent_{}_actor_output_{}_activation".format(self.agent_index, each), nn.Softmax())
                    ])))
            else:
                self.output_layer_list.append(nn.Sequential(
                    OrderedDict([
                        ("Agent_{}_actor_output_{}".format(self.agent_index, each), nn.Linear(self.units_num, shape)),
                    ])))

    def forward(self, input_list):
        x = torch.concatenate(input_list)
        for hidden_layer in self.hidden_layer_list:
            x = hidden_layer(x)
        output_list = []
        for output_layer in self.output_layer_list:
            output_list.append(output_layer(x))
        return output_list

if __name__ == "__main__":
    agent_index = 1
    state_shape = [7, 7, 7, 7, 7]
    action_shape = [6, 6, 6]
    critic_model = DDPG_Critic_Model(agent_index, state_shape, action_shape, 32, 5)
    actor_model = DDPG_Actor_Model(agent_index, state_shape, action_shape, 32, 5)
    print(critic_model)
    print(actor_model)
