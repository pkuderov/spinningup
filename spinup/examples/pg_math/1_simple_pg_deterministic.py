from contextlib import contextmanager
from typing import Tuple

import gym
import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Dense


@contextmanager
def managed_gym_environment(env_name: str, debug: bool):
    env = gym.make(env_name)
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


class VanillaPolicyGradientRL:
    def __init__(self, env_name: str, debug: bool = False):
        self.env_name = env_name
        self.debug = debug

    def run_train(self, hidden_layers: Tuple[int], epochs: int) -> None:
        with managed_gym_environment(self.env_name, self.debug) as env:
            observation_shape = env.observation_space.shape
            n_actions = env.action_space.n

            np.random.seed(42)
            tf.random.set_random_seed(42)
            env.seed(42)

            observation = Input(shape=observation_shape, dtype=tf.float32)
            hidden_layer = Dense(4, activation=tf.tanh)(observation)
            logits = Dense(n_actions)(hidden_layer)

            probs = tf.nn.softmax(logits)
            actions = tf.argmax(probs, axis=1)

            rewards_phi = tf.placeholder(shape=(None,), dtype=tf.float32)
            actions_phi = tf.placeholder(shape=(None,), dtype=tf.int32)
            action_indices = tf.stack(
                [tf.range(tf.size(actions_phi)), actions_phi],
                axis=-1
            )
            likelihood = tf.log(tf.gather_nd(probs, action_indices))
            loss = -tf.reduce_mean(likelihood * rewards_phi)

            train_op = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss)

            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())

            obs = env.reset()
            for i in range(10):
                logits_, probs_, actions_ = sess.run([logits, probs, actions], {observation: obs.reshape(1, -1)})
                action = actions_[0]
                print(logits_)
                print(probs_)
                print(action)

                obs, reward, done, _ = env.step(action)

                loss_, _ = sess.run([loss, train_op], {
                    observation: obs.reshape(1, -1),
                    rewards_phi: [reward],
                    actions_phi: [action]
                })
                print('loss:', loss_)

            input()

    def _train_one_epoch(self, env, batch_size, max_total_steps, render=True):
        observation = env.reset()
        done = False
        actions = []
        rewards = []

        while True:
            env.render()
            print(observation)

            # act in the environment
            act = np.random.randint(0, 2)
            observation, reward, done, _ = env.step(act)

            actions.append(act)
            rewards.append(reward)

            if done or len(rewards) > 100:
                print(actions)
                print(rewards)
                break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()

    debug = args.debug
    if not debug:
        print('\nUsing simplest formulation of policy gradient.\n')

    VanillaPolicyGradientRL(args.env_name, debug=debug).run_train(
        hidden_layers=(32,),
        epochs=1
    )
