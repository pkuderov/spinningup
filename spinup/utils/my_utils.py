from contextlib import contextmanager

import gym
from gym.spaces import Box, Discrete


@contextmanager
def managed_gym_environment(env_name: str, debug: bool, unlock: bool):
    env = gym.make(env_name)
    if unlock:
        env = env.env
    if debug:
        print('===> Env')

    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    try:
        yield env
    finally:
        env.close()
        if debug:
            print('<=== Env')