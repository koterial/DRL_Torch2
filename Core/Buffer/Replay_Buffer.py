import os
import copy
import pickle
import numpy as np


# 经验回放池
class Replay_Buffer():
    def __init__(self, *args, **kwargs):
        self.config = copy.deepcopy(kwargs)
        self.buffer_shape = int(self.config.get("shape", 1e5))
        state_shape = self.config["state_shape"]
        action_shape = self.config["action_shape"]
        if isinstance(state_shape, list): state_shape = tuple(state_shape)
        if isinstance(action_shape, list): action_shape = tuple(action_shape)
        self.state_buffer = np.zeros((self.buffer_shape, *state_shape), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_shape, *action_shape), dtype=np.float32)
        self.log_prob_buffer = np.zeros((self.buffer_shape, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_shape, *state_shape), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_shape, 1), dtype=np.float32)
        self.terminated_buffer = np.zeros((self.buffer_shape, 1), dtype=np.bool_)
        self.truncated_buffer = np.zeros((self.buffer_shape, 1), dtype=np.bool_)
        self.data_pointer = 0
        self.current_size = 0

    # 保存经验
    def remember(self, experience_batch):
        state_batch, action_batch, log_prob_batch, next_state_batch, reward_batch, terminated_batch, truncated_batch = map(np.asarray, zip(*experience_batch))
        batch_size = len(terminated_batch)
        indices = (np.arange(batch_size) + self.data_pointer) % self.buffer_shape
        self.state_buffer[indices] = state_batch
        self.action_buffer[indices] = action_batch
        self.log_prob_buffer[indices] = log_prob_batch.reshape(-1, 1) if log_prob_batch[0] is not None else 0 * np.ones((batch_size, 1))
        self.next_state_buffer[indices] = next_state_batch
        self.reward_buffer[indices] = reward_batch.reshape(-1, 1)
        self.terminated_buffer[indices] = terminated_batch.reshape(-1, 1)
        self.truncated_buffer[indices] = truncated_batch.reshape(-1, 1)
        self.data_pointer = (self.data_pointer + batch_size) % self.buffer_shape
        self.current_size = min(self.current_size + batch_size, self.buffer_shape)

    # 取出一批经验
    def sample(self, batch_shape):
        indices = np.random.randint(0, self.current_size, size=batch_shape)
        state_batch = self.state_buffer[indices]
        action_batch = self.action_buffer[indices]
        log_prob_batch = self.log_prob_buffer[indices]
        next_state_batch = self.next_state_buffer[indices]
        reward_batch = self.reward_buffer[indices]
        terminated_batch = self.terminated_buffer[indices]
        truncated_batch = self.truncated_buffer[indices]
        return state_batch, action_batch, log_prob_batch, next_state_batch, reward_batch, terminated_batch, truncated_batch, None, None

    # 返回经验回放池大小
    def size(self):
        return self.current_size

    # 重置经验回放池
    def reset(self):
        self.data_pointer = 0
        self.current_size = 0

    # 保存经验回放池
    def save(self, index, file_path):
        os.makedirs(file_path, exist_ok=True)
        buffer = {
            "state": self.state_buffer[:self.current_size],
            "action": self.action_buffer[:self.current_size],
            "log_prob": self.log_prob_buffer[:self.current_size],
            "next_state": self.next_state_buffer[:self.current_size],
            "reward": self.reward_buffer[:self.current_size],
            "terminated": self.terminated_buffer[:self.current_size],
            "truncated": self.truncated_buffer[:self.current_size],
            "data_pointer": self.data_pointer,
            "current_size": self.current_size,
        }
        with open(file_path + "/Agent_{}_Replay_Buffer_data.pickle".format(index), "wb") as f:
            pickle.dump(buffer, f)

    # 读取经验回放池
    def load(self, index, file_path):
        with open(file_path + "/Agent_{}_Replay_Buffer_data.pickle".format(index), "rb") as f:
            buffer = pickle.load(f)
        n = len(buffer["state"])
        self.state_buffer[:n] = buffer["state"]
        self.action_buffer[:n] = buffer["action"]
        self.log_prob_buffer[:n] = buffer["log_prob"]
        self.next_state_buffer[:n] = buffer["next_state"]
        self.reward_buffer[:n] = buffer["reward"]
        self.terminated_buffer[:n] = buffer["terminated"]
        self.truncated_buffer[:n] = buffer["truncated"]
        self.data_pointer = buffer.get("data_pointer", n % self.buffer_shape)
        self.current_size = buffer.get("current_size", n)