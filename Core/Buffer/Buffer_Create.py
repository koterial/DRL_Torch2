import copy
from Core.Buffer.Replay_Buffer import Replay_Buffer
from Core.Buffer.Prioritized_Replay_Buffer import Prioritized_Replay_Buffer


def buffer_create(buffer_config):
    init_kwargs = copy.deepcopy(buffer_config)
    buffer_class_name = init_kwargs.pop("class")
    if buffer_class_name == "Replay_Buffer":
        buffer = Replay_Buffer(**init_kwargs)
    elif buffer_class_name == "Prioritized_Replay_Buffer":
        buffer = Prioritized_Replay_Buffer(**init_kwargs)
    else:
        raise ValueError("未知的 Buffer 类别:", buffer_config['class'])
    return buffer