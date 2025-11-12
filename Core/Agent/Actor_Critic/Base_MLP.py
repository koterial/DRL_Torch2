import copy
import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional


class Base_MLP(nn.Module):
    """
    通用的 DRL MLP 基类 (支持多输入、多输出、独立激活), 可以作为 DRL 库中 Actor 和 Critic 的通用后端.

    --- 用例示例 ---

    1. V(s) Critic:
        - input_shape: [state_shape]
        - output_shape: [1]
        - activation: "linear"

    2. Q(s, a) Critic:
        - input_shape: [state_shape, action_shape]
        - output_shape: [1]
        - activation: "linear"

    3. Twin Q(s, a) Critic (TD3/SAC):
        - input_shape: [state_shape, action_shape]
        - output_shape: [1, 1]
        - activation: "linear"

    4. 确定性 Actor (DDPG/TD3):
        - input_shape: [state_shape]
        - output_shape: [action_shape]
        - activation: "tanh"

    5. 随机性高斯 Actor (SAC/A2C):
        - input_shape: [state_shape]
        - output_shape: [action_shape, action_shape] (mean, log_std)
        - activation: ["linear", "linear"] (或 ["tanh", "linear"])

    --- 配置参数 (kwargs) ---

    input_shape (Union[int, List[int]]):
        单个或多个输入维度.

    output_shape (Union[int, List[int]]):
        单个或多个输出维度.

    hidden_shape (List[int]):
        隐藏层的维度列表. 必须至少包含一个元素.

    activation (Union[str, List[str]]):
        输出层的激活函数.
        - "linear": (默认) 单个字符串, 应用于所有输出.
        - ["linear", "tanh"]: 字符串列表, 必须与 output_shape 长度匹配.

    hidden_activation (str, optional):
        隐藏层的激活函数 (例如: "relu", "tanh"). 默认为 "relu".
    """

    def __init__(self, *args, **kwargs):
        super(Base_MLP, self).__init__()
        self.config = copy.deepcopy(kwargs)

        # --- 1. 标准化输入/输出维度 ---
        raw_input_shape = self.config["input_shape"]
        if isinstance(raw_input_shape, int):
            self.input_shape = [raw_input_shape]
        elif isinstance(raw_input_shape, list):
            if any(isinstance(i, list) for i in raw_input_shape):
                self.input_shape = [item for sublist in raw_input_shape for item in (sublist if isinstance(sublist, list) else [sublist])]
            else:
                self.input_shape = raw_input_shape
        else:
            raise TypeError(f"input_shape 必须是 int 或 list, 但收到了 {type(raw_input_shape)}")

        raw_output_shape = self.config["output_shape"]
        if isinstance(raw_output_shape, int):
            self.output_shape = [raw_output_shape]
        elif isinstance(raw_output_shape, list):
            if any(isinstance(i, list) for i in raw_output_shape):
                self.output_shape = [item for sublist in raw_output_shape for item in (sublist if isinstance(sublist, list) else [sublist])]
            else:
                self.output_shape = raw_output_shape
        else:
            raise TypeError(f"output_shape 必须是 int 或 list, 但收到了 {type(raw_output_shape)}")

        self.hidden_shape: List[int] = self.config["hidden_shape"]
        if not self.hidden_shape:
            raise ValueError("hidden_shape 必须至少包含一个元素")

        # --- 2. 获取激活函数名称 ---
        self.hidden_activation_name = self.config.get("hidden_activation", "relu")

        # 默认使用 "linear", 因为它最通用 (V, Q, Logits 都用它)
        self.output_activation_config = self.config.get("activation", "linear")

        # --- 3. 标记, 用于优化forward ---
        self._is_single_input = (len(self.input_shape) == 1)
        self._is_single_output = (len(self.output_shape) == 1)

        self.model_create()

    def _get_activation(self, name: str) -> nn.Module:
        if name == "linear":
            return nn.Identity()
        elif name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "softmax":
            return nn.Softmax(dim=-1)
        else:
            raise ValueError(f"不支持的激活函数: {name}")

    def model_create(self):
        # --- 1. 创建隐藏层激活实例 ---
        self.hidden_activation_layer = self._get_activation(self.hidden_activation_name)

        # --- 2. 创建输出层激活实例 (ModuleList) ---
        self.output_activation_layers = nn.ModuleList()
        num_outputs = len(self.output_shape)
        activation_names: List[str] = []

        if isinstance(self.output_activation_config, str):
            name = self.output_activation_config
            activation_names = [name] * num_outputs

        elif isinstance(self.output_activation_config, list):
            if len(self.output_activation_config) != num_outputs:
                raise ValueError(
                    f"Config 'activation' (list) 长度 ({len(self.output_activation_config)}) "
                    f"必须匹配 'output_shape' 长度 ({num_outputs})"
                )
            activation_names = self.output_activation_config
        else:
            raise TypeError(
                f"Config 'activation' 必须是 str 或 List[str], "
                f"但收到了 {type(self.output_activation_config)}"
            )

        # 根据标准化的名称列表创建激活层
        for name in activation_names:
            self.output_activation_layers.append(self._get_activation(name))

        # --- 3. 计算总维度 ---
        total_input_shape = sum(self.input_shape)
        total_output_shape = sum(self.output_shape)  # 输出层仍然是总维度

        # --- 4. 创建网络层 (MLP) ---
        self.input_layer = nn.Linear(total_input_shape, self.hidden_shape[0])

        self.hidden_layer_list = nn.ModuleList()
        for i in range(len(self.hidden_shape) - 1):
            self.hidden_layer_list.append(
                nn.Linear(self.hidden_shape[i], self.hidden_shape[i + 1])
            )

        # 输出层 (Linear) 负责映射到总的输出维度
        self.output_layer = nn.Linear(self.hidden_shape[-1], total_output_shape)

    def forward(self, *inputs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        前向传播.

        流程:
        1. (多输入) 拼接
        2. 通过 MLP (input -> hidden -> output_linear)
        3. (多输出) 拆分
        4. (多输出) 分别激活
        """

        # --- 1. 处理输入: 拼接 ---
        if self._is_single_input:
            x = inputs[0]
        else:
            if len(inputs) != len(self.input_shape):
                raise ValueError(f"模型期望 {len(self.input_shape)} 个输入, 但收到了 {len(inputs)} 个.")
            try:
                x = torch.cat(inputs, dim=-1)
            except RuntimeError as e:
                print(f"输入张量拼接失败. 检查所有输入的Batch Size是否一致. Error: {e}")
                print(f"Inputs shapes: {[inp.shape for inp in inputs]}")
                raise

        # --- 2. 通过 MLP (直到最后一个线性层) ---
        x = self.input_layer(x)
        x = self.hidden_activation_layer(x)

        for hidden_layer in self.hidden_layer_list:
            x = hidden_layer(x)
            x = self.hidden_activation_layer(x)

        # 这是未激活的最终输出
        pre_activation_output = self.output_layer(x)

        # --- 3. 处理输出: 拆分 -> 激活 ---
        if self._is_single_output:
            # 优化: 单输出时, 直接激活并返回
            output = self.output_activation_layers[0](pre_activation_output)
            return output
        else:
            # 多输出:
            # 1. 按照 output_shape 拆分张量
            split_outputs = torch.split(pre_activation_output, self.output_shape, dim=-1)

            # 2. 对每个拆分后的张量应用其对应的激活函数
            activated_outputs = [
                self.output_activation_layers[i](split_outputs[i])
                for i in range(len(split_outputs))
            ]

            # 3. 返回元组
            return tuple(activated_outputs)


if __name__ == "__main__":
    state_shape = 10
    action_shape = 4
    hidden_shape = [256, 128]
    batch_size = 32

    # --- 1. 用作 V(s) Critic ---
    print("--- 1. V(s) Critic ---")
    v_config = {
        "input_shape": state_shape,
        "output_shape": 1,
        "hidden_shape": hidden_shape,
        "activation": "linear"  # 明确指定
    }
    v_critic = Base_MLP(**v_config)
    s = torch.randn(batch_size, state_shape)
    v_val = v_critic(s)
    print(f"V(s) Output shape: {v_val.shape}")
    print("-" * 20)

    # --- 2. 用作 Twin Q(s, a) Critic ---
    print("--- 2. Twin Q(s, a) Critic ---")
    twin_q_config = {
        "input_shape": [state_shape, action_shape],
        "output_shape": [1, 1],
        "hidden_shape": hidden_shape,
        "activation": "linear"
    }
    twin_q_critic = Base_MLP(**twin_q_config)
    s = torch.randn(batch_size, state_shape)
    a = torch.randn(batch_size, action_shape)
    q1, q2 = twin_q_critic(s, a)
    print(f"Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")
    print("-" * 20)

    # --- 3. 用作 确定性 Actor ---
    print("--- 3. 确定性 Actor ---")
    det_actor_config = {
        "input_shape": state_shape,
        "output_shape": action_shape,
        "hidden_shape": hidden_shape,
        "activation": "tanh"  # 覆盖默认激活
    }
    det_actor = Base_MLP(**det_actor_config)
    s = torch.randn(batch_size, state_shape)
    action = det_actor(s)
    print(f"action shape: {action.shape}")
    print(f"action value (tanh): {action[0]}")
    print("-" * 20)

    # --- 4. 用作 随机性高斯 Actor (mean, log_std) ---
    print("--- 4. 随机性高斯 Actor ---")
    stoch_actor_config = {
        "input_shape": state_shape,
        "output_shape": [action_shape, action_shape],
        "hidden_shape": hidden_shape,
        "activation": ["tanh", "linear"]  # 独立激活 (约束 mean, log_std 不约束)
    }
    stoch_actor = Base_MLP(**stoch_actor_config)
    s = torch.randn(batch_size, state_shape)
    mean, log_std = stoch_actor(s)
    print(f"Mean shape: {mean.shape}, Log_Std shape: {log_std.shape}")
    print(f"Mean (tanh): {mean[0]}")
    print(f"Log_Std (linear): {log_std[0]}")
    print("-" * 20)