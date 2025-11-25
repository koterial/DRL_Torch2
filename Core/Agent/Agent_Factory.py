import copy
import torch
import queue
import torch.multiprocessing as mp
from Core.Buffer.Buffer_Create import buffer_create


class Agent():
    def __init__(self, *args, **kwargs):
        # --- 1. 弹出所有非简单类型的对象 ---
        self.learner_class = kwargs.pop("learner_class")
        self.collector_class = kwargs.pop("collector_class")
        self.agent_config = kwargs.pop("agent_config")
        self.buffer_config = kwargs.pop("buffer_config")

        self.config = copy.deepcopy(kwargs)

        # --- 2. 创建单进程队列 ---
        experience_queue_size = self.config.get("experience_queue_size", 50000)
        self.experience_queue = queue.Queue(maxsize=experience_queue_size)
        self.weight_queue = queue.Queue(maxsize=1)
        self.all_weight_queues = [self.weight_queue]
        sample_queue_size = self.config.get("sample_queue_size", 10)
        self.sample_queue = mp.Queue(maxsize=sample_queue_size)
        self.error_queue = mp.Queue(maxsize=sample_queue_size)

        # --- 3. 确定运行的设备 ---
        self.learner_device = torch.device(self.agent_config.get("learner_device", "cuda"))
        self.collector_device = torch.device(self.agent_config.get("collector_device", "cpu"))

        # --- 4. 创建 Learner ---
        learner_config = copy.deepcopy(self.agent_config)
        learner_config["index"] = "learner_0"
        learner_config["mini_batch_shape"] = int(self.agent_config.get("batch_shape", 256))
        learner_config["all_weight_queues"] = self.all_weight_queues
        learner_config["sample_queue"] = self.sample_queue
        learner_config["error_queue"] = self.error_queue
        self.learner = self.learner_class(**learner_config)
        print(f"[{self.learner.index}] 启动在 {self.learner.device}")

        # --- 5. 创建 Collector ---
        collector_config = copy.deepcopy(self.agent_config)
        collector_config["index"] = "collector_0"
        collector_config["experience_queue"] = self.experience_queue
        collector_config["weight_queue"] = self.weight_queue
        self.collector = self.collector_class(**collector_config)
        print(f"[{self.collector.index}] 启动在 {self.collector.device}")

        self.learner._distribute_weights(device=self.collector_device)
        print(f"[{self.learner.index}] 已分发初始权重")
        self.collector._check_for_new_weights()
        print(f"[{self.collector.index}] 收到初始权重, 开始采集")

        # --- 6. 创建 Replay Buffer ---
        self.batch_shape = self.agent_config["batch_shape"]
        self.buffer_config["state_shape"] = self.agent_config["state_shape"]
        self.buffer_config["action_shape"] = self.agent_config["action_shape"]
        self.replay_buffer = buffer_create(self.buffer_config)
        self.prioritized_replay = True if self.buffer_config.get("class") == "Prioritized_Replay_Buffer" else False

    def get_action(self, state, deterministic=False):
        return self.collector.get_action(state, deterministic)

    def remember(self, experience):
        self.collector.remember(experience)
        try:
            experience_batch = self.experience_queue.get_nowait()
            self.replay_buffer.remember(experience_batch)
        except queue.Empty:
            pass

    def train(self):
        if self.replay_buffer.size() > self.batch_shape and not self.sample_queue.full():
            batch = self.replay_buffer.sample(self.batch_shape)
            if batch[0] is not None:
                self.sample_queue.put(batch)
        self.learner.train()
        if self.prioritized_replay and not self.error_queue.empty():
            try:
                index_batch, error_batch = self.error_queue.get_nowait()
                self.replay_buffer.batch_update(index_batch, error_batch)
            except queue.Empty:
                pass

    def model_save(self, file_path):
        self.learner.model_save(file_path)

    def model_load(self, file_path, index=None):
        self.learner.model_load(file_path, index)
        self.learner._distribute_weights(device=self.collector_device)


class Factory():
    def __init__(self, *args, **kwargs):
        # --- 1. 弹出所有非简单类型的对象 ---
        self.learner_class = kwargs.pop("learner_class")
        self.collector_class = kwargs.pop("collector_class")
        self.agent_config = kwargs.pop("agent_config")
        self.buffer_config = kwargs.pop("buffer_config")

        self.config = copy.deepcopy(kwargs)

        # --- 2. 获取配置 ---
        self.num_learners = self.config.get("num_learners", 1)
        self.num_collectors = self.config.get("num_collectors", 1)

        # --- 3. 创建多进程队列 ---
        experience_queue_size = self.config.get("experience_queue_size", 50000)
        self.experience_queue = mp.Queue(maxsize=experience_queue_size)
        self.all_weight_queues = [mp.Queue(maxsize=1) for _ in range(self.num_collectors)]
        sample_queue_size = self.config.get("sample_queue_size", 10)
        self.sample_queue = mp.Queue(maxsize=sample_queue_size)
        self.error_queue = mp.Queue(maxsize=sample_queue_size)

        # --- 4. 确定运行的设备 ---
        self.learner_device = torch.device(self.agent_config.get("learner_device", "cuda"))
        self.collector_device = torch.device(self.agent_config.get("collector_device", "cpu"))

        # --- 5. 只创建Learner和Collector配置, 而不是实例 ---
        self.learner_configs = []
        for i in range(self.num_learners):
            learner_config = copy.deepcopy(self.agent_config)
            learner_config["index"] = f"learner_{i}"
            learner_config["mini_batch_shape"] = int(self.agent_config.get("batch_shape", 256) / 1)
            learner_config["all_weight_queues"] = self.all_weight_queues
            learner_config["sample_queue"] = self.sample_queue
            learner_config["error_queue"] = self.error_queue
            self.learner_configs.append(learner_config)

        self.collector_configs = []
        for i in range(self.num_collectors):
            collector_config = copy.deepcopy(self.agent_config)
            collector_config["index"] = f"collector_{i}"
            collector_config["experience_queue"] = self.experience_queue
            collector_config["weight_queue"] = self.all_weight_queues[i]
            self.collector_configs.append(collector_config)

        # --- 6. 创建 Replay Buffer ---
        self.buffer_config["state_shape"] = self.agent_config["state_shape"]
        self.buffer_config["action_shape"] = self.agent_config["action_shape"]
        self.replay_buffer = buffer_create(self.buffer_config)
        self.prioritized_replay = True if self.buffer_config.get("class") == "Prioritized_Replay_Buffer" else False

    def get_learner_config(self, index=0):
        return self.learner_configs[index]

    def get_collector_config(self, index=0):
        return self.collector_configs[index]

    def get_learner_class(self):
        return self.learner_class

    def get_collector_class(self):
        return self.collector_class

    def get_experience_queue(self):
        return self.experience_queue

    def get_sample_queue(self):
        return self.sample_queue

    def get_error_queue(self):
        return self.error_queue