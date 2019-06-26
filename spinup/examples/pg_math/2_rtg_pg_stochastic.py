from contextlib import contextmanager
from timeit import default_timer as timer
from typing import Tuple

import gym
import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box
from tensorflow.python.keras import Input
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
    class _Model:
        def __init__(
                self, observation_shape: Tuple[int, ...], n_actions: int,
                hidden_layers: Tuple[int, ...],
                learning_rate: float = 1e-2, stochasticity: float = .1
        ):
            self._n_actions = n_actions
            self._stochasticity = stochasticity

            observation = Input(shape=observation_shape, dtype=tf.float32)
            x = observation
            for hidden_layer in hidden_layers:
                x = Dense(hidden_layer, activation=tf.nn.relu)(x)
            logits = Dense(n_actions)(x)
            actions = tf.squeeze(tf.random.categorical(logits=logits, num_samples=1), axis=1)

            rewards_phi = tf.placeholder(shape=(None,), dtype=tf.float32)
            actions_phi = tf.placeholder(shape=(None,), dtype=tf.int32)
            actions_mask = tf.one_hot(actions_phi, n_actions)
            log_probs = tf.nn.log_softmax(logits)
            likelihood = tf.reduce_sum(actions_mask * log_probs, axis=1)
            loss = -tf.reduce_mean(rewards_phi * likelihood)

            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

            self._session = tf.InteractiveSession()
            self._session.run(tf.global_variables_initializer())

            self._observation = observation
            self._actions = actions
            self._rewards_phi = rewards_phi
            self._actions_phi = actions_phi
            self._loss = loss
            self._train_op = train_op

        def train_one_epoch(self, observations, actions, rewards):
            loss, _ = self._session.run(
                [self._loss, self._train_op],
                feed_dict={
                    self._observation: observations,
                    self._actions_phi: actions,
                    self._rewards_phi: rewards,
                }
            )
            return loss

        def get_loss(self, observations, actions, rewards):
            loss = self._session.run(
                self._loss,
                feed_dict={
                    self._observation: observations,
                    self._actions_phi: actions,
                    self._rewards_phi: rewards,
                }
            )
            return loss

        def predict_action(self, observations):
            actions_batch = self._session.run(self._actions, {
                self._observation: observations.reshape(1, -1)
            })
            action = actions_batch[0]
            if np.random.rand() < self._stochasticity:
                # action = self._sample_another_action(action)
                action = np.random.randint(self._n_actions)
            return action

        def _sample_another_action(self, action):
            a = np.random.randint(self._n_actions - 1)
            return a if a < action else a + 1

    def __init__(self, env_name: str, debug: bool = False):
        self.env_name = env_name
        self.debug = debug

    def run_train(self,
            hidden_layers: Tuple[int], epochs: int,
            epoch_episodes: int, epoch_steps: int,
            learning_rate: float, render: bool
    ) -> None:
        with managed_gym_environment(self.env_name, self.debug) as env:
            observation_shape = env.observation_space.shape
            n_actions = env.action_space.n

            model = self._Model(
                observation_shape, n_actions,
                hidden_layers=hidden_layers,
                learning_rate=learning_rate,
                stochasticity=.99
            )
            for epoch in range(epochs):
                observations, actions, rewards, episode_scores, episode_lengths = self._get_one_epoch_samples(
                    env, model, n_episodes=epoch_episodes, max_total_steps=epoch_steps,
                    render=render
                )

                if episode_scores.shape[0] == 0:
                    continue

                t = timer()
                loss = model.train_one_epoch(
                    observations=observations,
                    actions=actions,
                    rewards=rewards
                )
                dt = 1e7 * (timer() - t) / observations.shape[0]

                model._stochasticity *= .0

                mean_score = episode_scores.mean()
                mean_length = episode_lengths.mean()
                print(
                    f'[{epoch}]: loss: {loss:.3f}    score: {mean_score:.3f}    length: {mean_length:.3f}  t: {dt:.4f}'
                )

            if render:
                input()

    def _get_one_epoch_samples(
            self, env, model: 'VanillaPolicyGradientRL._Model',
            n_episodes: int, max_total_steps: int,
            render: bool = True
    ):
        actions = np.ones(shape=(max_total_steps,), dtype=np.int32)
        rewards = np.ones(shape=(max_total_steps,), dtype=np.float32)
        current_rewards = np.ones(shape=(max_total_steps,), dtype=np.float32)
        episode_scores = np.ones(shape=(n_episodes,), dtype=np.float32)
        episode_lengths = np.ones(shape=(n_episodes,), dtype=np.int32)

        step, episode, done, ep_step = 0, 0, False, 0
        observation = env.reset()
        observations_shape = (max_total_steps,) + observation.shape
        observations = np.ones(shape=observations_shape, dtype=np.float32)

        while step < max_total_steps and episode < n_episodes:
            if episode == 0 and render:
                env.render()

            observations[step] = observation.copy()

            action = model.predict_action(observation)
            actions[step] = action

            observation, reward, done, _ = env.step(action)
            current_rewards[ep_step] = reward

            ep_step += 1
            step += 1
            if done:
                observation = env.reset()
                score = current_rewards[:ep_step].sum()
                rewards[(step - ep_step):step] = self._reward_to_go(current_rewards[:ep_step])
                episode_scores[episode] = score
                episode_lengths[episode] = ep_step

                episode += 1
                ep_step = 0

        observations = observations[:step-ep_step]
        actions = actions[:step-ep_step]
        rewards = rewards[:step-ep_step]
        episode_scores = episode_scores[:episode]
        episode_lengths = episode_lengths[:episode]

        return observations, actions, rewards, episode_scores, episode_lengths

    def _reward_to_go(self, rewards):
        return np.cumsum(rewards[::-1])[::-1]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--epoch_episodes', type=int, default=100)
    parser.add_argument('--epoch_steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    debug = args.debug
    if not debug:
        print('\nUsing simplest formulation of policy gradient.\n')

    VanillaPolicyGradientRL(args.env_name, debug=debug).run_train(
        hidden_layers=(32,),
        epochs=args.epochs,
        epoch_episodes=args.epoch_episodes,
        epoch_steps=args.epoch_steps,
        learning_rate=args.lr,
        render=args.render
    )
