import copy
import numpy as np
import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional


class Base_MLP(nn.Module):
    """
    通用的 DRL MLP 基类 (支持多输入、多输出、独立激活), 可以作为 DRL 库中 Actor 和 Critic 的通用后端.
    input_shape 和 output_shape 均支持 int, List[int], Tuple[N, M] 的任意嵌套组合.

    --- 用例示例 ---

    1. V(s) Critic:
        - input_shape: state_shape (int)
        - output_shape: 1 (int)
        - activation: "linear"

    2. Q(s, a) Critic:
        - input_shape: [state_shape, action_shape] (List[Union[int, List, Tuple]])
        - output_shape: 1 (int)
        - activation: "linear"

    3. Twin Q(s, a) Critic (TD3/SAC):
        - input_shape: [state_shape, action_shape] (List[Union[int, List, Tuple]])
        - output_shape: (2, 1) (Tuple[int, int])/[1, 1] (List[int])
        - activation: "linear"

    4. 确定性 Actor (DDPG/TD3):
        - input_shape: state_shape (int)
        - output_shape: action_shape (int)
        - activation: "tanh"

    5. 随机性高斯 Actor (SAC/A2C):
        - input_shape: state_shape (int)
        - output_shape: (2, action_shape) (Tuple[int, int])/[action_shape, action_shape] (List[int])
        - activation: "linear"/["linear", "linear"]

    --- 配置参数 (kwargs) ---

    input_shape (Union[int, List[int]]):
        单个或多个输出维度.
        - int: 4 (单头, 维度 4)
        - List[int]: [4, 1] (两个输出头, 维度 4 和 1)
        - Tuple[int, int]: (2, 4) (N=2, X=4, 2个输出头, 维度均为 4)

    output_shape (Union[int, List[int], Tuple[int, int]]): [核心修改]
        单个或多个输出维度.
        - int: 4 (单个输出头, 维度 4)
        - List[int]: [4, 1] (两个输出头, 维度 4 和 1)
        - Tuple[int, int]: (2, 4) (N=2, X=4, 2个输出头, 维度均为 4)


    hidden_shape (List[int]):
        隐藏层的维度列表. 必须至少包含一个元素.

    activation (Union[str, List[str]]):
        输出层的激活函数.
        - "linear": (默认) 单个字符串, 应用于所有输出.
        - ["linear", "tanh"]: 字符串列表, 必须与 num_output_heads 长度匹配.

    hidden_activation (str, optional):
        隐藏层的激活函数 (例如: "relu", "tanh"). 默认为 "relu".
    """

    def __init__(self, *args, **kwargs):
        super(Base_MLP, self).__init__()
        self.config = copy.deepcopy(kwargs)

        # --- 1. 标准化输入/输出维度 ---
        self.input_shape = self._flatten_shape_config(self.config["input_shape"])
        self.output_shape = self._flatten_shape_config(self.config["output_shape"])
        if not self.input_shape:
            raise ValueError("input_shape 在展平后不能为空")
        if not self.output_shape:
            raise ValueError("output_shape 在展平后不能为空")

        self.hidden_shape: List[int] = self.config["hidden_shape"]
        if not self.hidden_shape:
            raise ValueError("hidden_shape 必须至少包含一个元素")

        # --- 2. 获取激活函数名称 ---
        self.hidden_activation_name = self.config.get("hidden_activation", "relu")
        self.output_activation_config = self.config.get("activation", "linear")

        # --- 3. 标记, 用于优化forward ---
        self._is_single_input = (len(self.input_shape) == 1)
        self._is_single_output = (len(self.output_shape) == 1)

        self.model_create()

    # 递归地将 (int, List, Tuple) 的任意组合展平为 List[int]
    def _flatten_shape_config(self, shape_config) -> List[int]:
        if isinstance(shape_config, int):
            # 案例 1: 10 -> [10]
            return [shape_config]
        if isinstance(shape_config, list):
            # 案例 2: [10, 4] -> [10, 4]
            # 案例 3: [[10], [4]] -> [10, 4]
            # 案例 4: [10, (38,)] -> [10, 38]
            flattened_list = []
            for item in shape_config:
                flattened_list.extend(self._flatten_shape_config(item))
            return flattened_list
        if isinstance(shape_config, tuple):
            # 案例 5: (N, M) 格式, e.g., (2, 4) 或 (2, [1, 4])
            if (len(shape_config) == 2 and isinstance(shape_config[0], int) and shape_config[0] > 0 and isinstance(shape_config[1], (int, list, tuple))):
                num_heads = shape_config[0]
                head_config = shape_config[1]
                # 递归展平子配置 (e.g., [1, 4])
                flattened_head = self._flatten_shape_config(head_config)
                # 将子配置重复 N 次
                return flattened_head * num_heads
            # 案例 6: Shape 元组, e.g., (38,) 或 (84, 84, 3)
            else:
                try:
                    # 确保所有元素都是 int
                    if not all(isinstance(i, int) for i in shape_config):
                        raise TypeError(f"Shape tuple {shape_config} 包含 non-int 元素.")
                    if not shape_config:  # 处理空元组 ()
                        return []
                    # 展平: (38,) -> [38]
                    # 展平: (84, 84, 3) -> [21168]
                    return [int(np.prod(shape_config))]
                except Exception as e:
                    raise TypeError(f"shape (tuple) {shape_config} 既不是 (N, M) 格式, 也不是有效的 shape tuple. Error: {e}")

        raise TypeError(f"shape 必须是 int, List, 或 Tuple, 但收到了 {type(shape_config)}")

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
        for name in activation_names:
            self.output_activation_layers.append(self._get_activation(name))

        # --- 3. 计算总维度 ---
        total_input_shape = sum(self.input_shape)
        total_output_shape = sum(self.output_shape)

        # --- 4. 创建网络层 (MLP) ---
        self.input_layer = nn.Linear(total_input_shape, self.hidden_shape[0])
        self.hidden_layer_list = nn.ModuleList()
        for i in range(len(self.hidden_shape) - 1):
            self.hidden_layer_list.append(nn.Linear(self.hidden_shape[i], self.hidden_shape[i + 1]))
        # 输出层负责映射到总的输出维度
        self.output_layer = nn.Linear(self.hidden_shape[-1], total_output_shape)

    def forward(self, *inputs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # --- 1. 处理输入: 拼接 ---
        if self._is_single_input:
            x = inputs[0]
        else:
            if len(inputs) != len(self.input_shape):
                raise ValueError(f"模型期望 {len(self.input_shape)} 个输入 (来自 {self.config['input_shape']} 展平后), 但收到了 {len(inputs)} 个.")
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
        pre_activation_output = self.output_layer(x)

        # --- 3. 处理输出: 拆分 -> 激活 ---
        if self._is_single_output:
            output = self.output_activation_layers[0](pre_activation_output)
            return output
        else:
            split_outputs = torch.split(pre_activation_output, self.output_shape, dim=-1)
            activated_outputs = [self.output_activation_layers[i](split_outputs[i]) for i in range(len(split_outputs))]
            return tuple(activated_outputs)


if __name__ == "__main__":
    state_shape = 10
    action_shape = 4
    hidden_shape = [256, 128]
    batch_size = 32

    # --- 1. V(s) Critic ---
    print("--- 1. V(s) Critic ---")
    v_config = {
        "input_shape": state_shape,
        "output_shape": 1,
        "hidden_shape": hidden_shape
    }
    v_critic = Base_MLP(**v_config)
    s = torch.randn(batch_size, state_shape)
    v_val = v_critic(s)
    print(f"V(s) Output shape: {v_val.shape}")
    print("-" * 20)

    # --- 2. Q(s, a) Critic (List[int]) ---
    print("--- 2. Q(s, a) Critic (input=List) ---")
    q_config = {
        "input_shape": [state_shape, action_shape],
        "output_shape": 1,
        "hidden_shape": hidden_shape,
    }
    q_critic = Base_MLP(**q_config)
    s = torch.randn(batch_size, state_shape)
    a = torch.randn(batch_size, action_shape)
    q_val = q_critic(s, a)
    print(f"Q(s,a) Output shape: {q_val.shape}")
    print("-" * 20)

    # --- 3. Twin Q(s, a) Critic ---
    print("--- 3. Twin Q(s, a) Critic ---")
    twin_q_config = {
        "input_shape": [state_shape, action_shape],
        "output_shape": [1, 1],
        "hidden_shape": hidden_shape,
    }
    twin_q_critic = Base_MLP(**twin_q_config)
    s = torch.randn(batch_size, state_shape)
    a = torch.randn(batch_size, action_shape)
    q1, q2 = twin_q_critic(s, a)
    print(f"Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")
    print("-" * 20)

    # --- 4. 确定性 Actor ---
    print("--- 4. 确定性 Actor ---")
    deterministic_actor_config = {
        "input_shape": state_shape,
        "output_shape": action_shape,
        "hidden_shape": hidden_shape,
        "activation": "tanh"
    }
    deterministic_actor = Base_MLP(**deterministic_actor_config)
    s = torch.randn(batch_size, state_shape)
    action = deterministic_actor(s)
    print(f"action shape: {action.shape}")
    print(f"action value (tanh): {action[0]}")
    print("-" * 20)

    # --- 4. 随机性高斯 Actor (mean, log_std) ---
    print("--- 4. 随机性高斯 Actor ---")
    gaussian_actor_config = {
        "input_shape": state_shape,
        "output_shape": [action_shape, action_shape],
        # "output_shape": (2, action_shape),
        "hidden_shape": hidden_shape,
        "activation": ["tanh", "linear"]  # 独立激活 (约束 mean, log_std 不约束)
    }
    gaussian_actor = Base_MLP(**gaussian_actor_config)
    s = torch.randn(batch_size, state_shape)
    mean, log_std = gaussian_actor(s)
    print(f"Mean shape: {mean.shape}, Log_Std shape: {log_std.shape}")
    print(f"Mean (tanh): {mean[0]}")
    print(f"Log_Std (linear): {log_std[0]}")
    print("-" * 20)

    # --- 4. 复杂嵌套输入输出 ---
    print("--- 4. 复杂嵌套输入输出 ---")
    complex_out_config = {
        "input_shape": (state_shape,),
        "output_shape": (2, [2, action_shape]),
        # "output_shape": (2, (2, action_shape)),
        "hidden_shape": hidden_shape,
        "activation": ["linear", "relu", "linear", "tanh"]  # 匹配 4 个头
    }
    complex_out_net = Base_MLP(**complex_out_config)
    s = torch.randn(batch_size, state_shape)
    v1, a1, v2, a2 = complex_out_net(s)
    print(f"Input shape: {s.shape}")
    print(f"V1: {v1.shape}, A1: {a1.shape}, V2: {v2.shape}, A2: {a2.shape}")
    print("-" * 20)