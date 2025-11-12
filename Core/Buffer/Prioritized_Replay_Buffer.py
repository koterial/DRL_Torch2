import os
import copy
import pickle
import numpy as np
from Core.Buffer.Segment_Tree import Sum_Tree


# 优先经验回放池
class Prioritized_Replay_Buffer():
    def __init__(self, *args, **kwargs):
        self.config = copy.deepcopy(kwargs)
        self.buffer_shape = int(self.config.get("buffer_shape", 1e5))
        self.alpha = self.config.get("alpha", 0.6)
        self.beta = self.config.get("beta", 0.4)
        self.beta_increase = self.config.get("beta_increase", 1e-3)
        self.min_priority = self.config.get("min_priority", 0.01)
        self.max_priority = self.config.get("max_priority", 1)
        self.sum_tree = Sum_Tree(self.buffer_shape)

    # 保存经验
    def remember(self, experience_batch):
        max_priority = np.max(self.sum_tree.tree[-self.sum_tree.capacity:])
        if max_priority == 0:
            max_priority = self.max_priority
        for experience in experience_batch:
            self.sum_tree.add(max_priority, experience)

    # 取出一批经验
    def sample(self, batch_shape):
        index_batch = np.zeros((batch_shape,), dtype=np.int32)
        weight_batch = np.zeros((batch_shape,))
        experiences = []
        priority_segment = self.sum_tree.total_p() / batch_shape
        self.beta = np.min([1., self.beta + self.beta_increase])
        if self.sum_tree.full_tree:
            min_prob = np.min(self.sum_tree.tree[-self.sum_tree.capacity:]) / self.sum_tree.total_p()
        else:
            min_prob = np.min(self.sum_tree.tree[-self.sum_tree.capacity:self.sum_tree.capacity + self.sum_tree.data_pointer - 1]) / self.sum_tree.total_p()
        min_prob = max(min_prob, 1e-6)
        for each in range(batch_shape):
            a, b = priority_segment * each, priority_segment * (each + 1)
            v = np.random.uniform(a, b)
            index, priority, experience = self.sum_tree.get_leaf(v)
            prob = priority / self.sum_tree.total_p()
            index_batch[each] = index
            weight_batch[each] = np.power(prob / min_prob, -self.beta)
            experiences.append(experience)
        state_batch, action_batch, log_prob_batch, next_state_batch, reward_batch, terminated_batch, truncated_batch = map(np.asarray, zip(*experiences))
        return state_batch, action_batch, log_prob_batch.reshape(-1, 1), next_state_batch, reward_batch.reshape(-1, 1), terminated_batch.reshape(-1, 1), truncated_batch.reshape(-1, 1), index_batch, weight_batch

    # 更新经验权重
    def batch_update(self, tree_index_batch, TD_error_batch):
        TD_error_batch += self.min_priority
        TD_error_batch = np.minimum(TD_error_batch, self.max_priority)
        priority_batch = np.power(TD_error_batch, self.alpha)
        for tree_index, priority in zip(tree_index_batch, priority_batch):
            self.sum_tree.update(tree_index, priority)

    # 返回经验回放池大小
    def size(self):
        if self.sum_tree.full_tree:
            return self.sum_tree.capacity
        else:
            return self.sum_tree.data_pointer

    # 重置经验回放池
    def reset(self):
        self.sum_tree = Sum_Tree(self.buffer_shape)

    # 保存经验回放池
    def save(self, index, file_path):
        os.makedirs(file_path, exist_ok=True)
        with open(file_path + "/Agent_{}_Replay_Buffer_weight.pickle".format(index), "wb") as f:
            pickle.dump(self.sum_tree.tree, f)
        with open(file_path + "/Agent_{}_Replay_Buffer_data.pickle".format(index), "wb") as f:
            pickle.dump(self.sum_tree.data, f)

    # 读取经验回放池
    def load(self, index, file_path):
        with open(file_path + "/Agent_{}_Replay_Buffer_data.pickle".format(index), "rb") as f:
            data = pickle.load(f)
        if len(data) != self.buffer_shape:
            print("智能体" + str(index) + "经验池不匹配")
        else:
            with open(file_path + "/Agent_{}_Replay_Buffer_weight.pickle".format(index), "rb") as f:
                tree = pickle.load(f)
            self.sum_tree.tree = tree
            self.sum_tree.data = data
            self.sum_tree.full_tree = True