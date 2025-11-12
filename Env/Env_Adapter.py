import numpy as np

def env_adapter(env_name, state, action, log_prob, next_state, reward, terminated, truncated):
    if env_name == "CartPole-v1":
        if terminated:
            reward -= 100
    elif env_name == "MountainCar-v0":
        pass
    elif env_name == "MountainCarContinuous-v0":
        reward += next_state[0] - 0.5
    elif env_name == "Pendulum-v1":
        reward = (reward + 8) / 8
        terminated = truncated
    elif env_name == "LunarLanderContinuous-v2":
        pass
    elif env_name == "BipedalWalker-v3" or env_name == "BipedalWalkerHardcore-v3":
        if reward <= -100:
            reward = -1
            terminated = True
        pass
    return state, action, log_prob, next_state, reward, terminated, truncated

def action_adapter(env_name, action):
    act = action
    if env_name == "CartPole-v1":
        act = act[0]
    elif env_name == "MountainCar-v0":
        act = act[0]
    elif env_name == "MountainCarContinuous-v0":
        pass
    elif env_name == "Pendulum-v1":
        act = act * 2
    elif env_name == "LunarLanderContinuous-v2":
        pass
    elif env_name == "BipedalWalker-v3" or env_name == "BipedalWalkerHardcore-v3":
        pass
    return act, action