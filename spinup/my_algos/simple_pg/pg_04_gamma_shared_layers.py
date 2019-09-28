from timeit import default_timer as timer
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense

from spinup.utils.clr import cyclic_learning_rate as clr
from spinup.utils.experiment import ExperimentResult, ExperimentInfo, EpochResult
from spinup.utils.my_utils import managed_gym_environment


class VanillaPolicyGradientRL:
    class _Model:
        def __init__(
                self, observation_shape: Tuple[int, ...], n_actions: int,
                hidden_layers: Tuple[int, ...],
                learning_rate: float = 1e-2, stochasticity: float = .1
        ):
            self._n_actions = n_actions
            self._stochasticity = stochasticity

            # reset default graph
            tf.reset_default_graph()

            # shared hidden part of NN
            self._observation = Input(shape=observation_shape, dtype=tf.float32)
            hidden_NN = self._get_hidden_layers(hidden_layers, self._observation)

            # Policy approximate
            logits, self._actions = self._get_policy_approx(hidden_NN, n_actions)

            # Value-function approximate
            self._value_func_pred = self._get_value_func_approx(hidden_NN)

            # Policy loss
            self._actions_phi = tf.placeholder(shape=(None,), dtype=tf.int32)
            self._rewards_phi = tf.placeholder(shape=(None,), dtype=tf.float32)
            self._policy_loss = self._get_policy_loss(
                n_actions, self._rewards_phi, self._actions_phi, logits, self._value_func_pred
            )

            # Value-function loss
            self._value_func_loss = self._get_value_func_loss(self._rewards_phi, self._value_func_pred)

            # Policy learning rate and optimizer
            self._policy_global_step = tf.Variable(0, trainable=False)
            self._policy_learning_rate, self._policy_train_op = self._get_optimizer(
                learning_rate, self._policy_global_step, self._policy_loss
            )

            # Value-function learning rate and optimizer
            self._value_func_global_step = tf.Variable(0, trainable=False)
            self._value_func_learning_rate, self._value_func_train_op = self._get_optimizer(
                learning_rate, self._value_func_global_step, self._value_func_loss
            )

            self._session = tf.InteractiveSession()
            self._session.run(tf.global_variables_initializer())

        @staticmethod
        def _get_hidden_layers(
                hidden_layers: Tuple[int, ...], observation: tf.Variable
        ):
            x = observation
            for hidden_layer in hidden_layers:
                x = Dense(hidden_layer, activation=tf.nn.relu)(x)
            return x

        @staticmethod
        def _get_policy_approx(hidden_layers: tf.Tensor, n_actions: int):
            logits = Dense(n_actions)(hidden_layers)
            actions = tf.squeeze(tf.random.categorical(logits=logits, num_samples=1), axis=1)
            return logits, actions

        @staticmethod
        def _get_value_func_approx(hidden_layers: tf.Tensor):
            value_pred = tf.squeeze(Dense(1)(hidden_layers), axis=1)
            return value_pred

        @staticmethod
        def _get_value_func_loss(rewards_phi: tf.Variable, value_pred: tf.Variable):
            loss = tf.losses.mean_squared_error(rewards_phi, value_pred)
            return loss

        @staticmethod
        def _get_policy_loss(
                n_actions: int, rewards_phi: tf.Variable, actions_phi: tf.Variable, logits: tf.Tensor,
                baseline: tf.Tensor
        ):
            actions_mask = tf.one_hot(actions_phi, n_actions)
            log_probs = tf.nn.log_softmax(logits)
            likelihood = tf.reduce_sum(actions_mask * log_probs, axis=1)
            rewards = rewards_phi - baseline
            loss = -tf.reduce_mean(likelihood * rewards)
            return loss

        @staticmethod
        def _get_optimizer(learning_rate: float, global_step: tf.Variable, loss: tf.Tensor):
            lr = clr(
                global_step=global_step,
                step_size=10,
                learning_rate=(learning_rate, learning_rate * 50),
                const_lr_decay=.5,
                max_lr_decay=.7
            )
            train_op = tf.train.AdamOptimizer(
                learning_rate=lr
            ).minimize(
                loss, global_step=global_step
            )
            return lr, train_op

        def train_one_epoch(self, observations, actions, rewards):
            value_func_loss, _, value_func_learning_rate = self._session.run(
                [self._value_func_loss, self._value_func_train_op, self._value_func_learning_rate],
                feed_dict={
                    self._observation: observations,
                    self._rewards_phi: rewards,
                }
            )
            policy_loss, _, policy_learning_rate = self._session.run(
                [self._policy_loss, self._policy_train_op, self._policy_learning_rate,],
                feed_dict={
                    self._observation: observations,
                    self._actions_phi: actions,
                    self._rewards_phi: rewards,
                }
            )
            return policy_loss, policy_learning_rate, value_func_loss, value_func_learning_rate

        def get_loss(self, observations, actions, rewards):
            loss = self._session.run(
                self._policy_loss,
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

        def close_sessions(self):
            self._session.close()

        def _sample_another_action(self, action):
            a = np.random.randint(self._n_actions - 1)
            return a if a < action else a + 1

    class _Sampler:
        def __init__(self, env, n_episodes, episode_steps, gamma, render: bool = True):
            self._env = env
            self._n_episodes = n_episodes
            self._episode_steps = episode_steps
            self._gamma = gamma
            self._render = render

            max_total_steps = episode_steps * n_episodes
            self._actions = np.empty(shape=(max_total_steps,), dtype=np.int32)
            self._rewards = np.empty(shape=(max_total_steps,), dtype=np.float32)
            self._current_rewards = np.empty(shape=(max_total_steps,), dtype=np.float32)
            self._episode_scores = np.empty(shape=(n_episodes,), dtype=np.float32)
            self._episode_lengths = np.empty(shape=(n_episodes,), dtype=np.int32)

            observation = self._env.reset()
            observations_shape = (max_total_steps,) + observation.shape
            self._observations = np.empty(shape=observations_shape, dtype=np.float32)

        def get_one_epoch_samples(self, model: 'VanillaPolicyGradientRL._Model'):
            step, episode, done, ep_step = 0, 0, False, 0
            observation = self._env.reset()

            while episode < self._n_episodes:
                if episode == 0 and self._render:
                    self._env.render()

                self._observations[step] = observation.copy()

                action = model.predict_action(observation)
                self._actions[step] = action

                observation, reward, done, _ = self._env.step(action)
                self._current_rewards[ep_step] = reward

                ep_step += 1
                step += 1
                if done or ep_step >= self._episode_steps:
                    observation = self._env.reset()
                    score = self._current_rewards[:ep_step].sum()
                    self._rewards[(step - ep_step):step] = self._reward_to_go(
                        self._current_rewards[:ep_step]
                    )
                    self._episode_scores[episode] = score
                    self._episode_lengths[episode] = ep_step

                    episode += 1
                    ep_step = 0

            observations = self._observations[:step - ep_step]
            actions = self._actions[:step - ep_step]
            rewards = self._rewards[:step - ep_step]
            episode_scores = self._episode_scores[:episode]
            episode_lengths = self._episode_lengths[:episode]

            return observations, actions, rewards, episode_scores, episode_lengths

        def _reward_to_go(self, rewards):
            result = rewards.copy()
            for i in range(result.shape[0] - 2, -1, -1):
                result[i] += self._gamma * result[i+1]
            return result

    def __init__(self, env_name: str, _debug: bool = False, unlock_env: bool = False):
        self.env_name = env_name
        self.debug = _debug
        self.unlock_env = unlock_env

    def run_train(
            self,
            hidden_layers: Tuple[int], epochs: int,
            epoch_episodes: int, episode_steps: int,
            learning_rate: float, gamma: float,
            render: bool, print_scores: bool
    ) -> ExperimentResult:
        with managed_gym_environment(self.env_name, self.debug, self.unlock_env) as env:
            observation_shape = env.observation_space.shape
            n_actions = env.action_space.n

            experiment_info = ExperimentInfo(env=self.env_name, epochs=epochs, hidden_layers=hidden_layers)
            experiment_result = ExperimentResult(experiment_info)

            model = self._Model(
                observation_shape, n_actions,
                hidden_layers=hidden_layers,
                learning_rate=learning_rate,
                stochasticity=.0
            )
            sampler = self._Sampler(
                env=env,
                n_episodes=epoch_episodes,
                episode_steps=episode_steps,
                gamma=gamma,
                render=render
            )
            for epoch in range(epochs):
                observations, actions, rewards, episode_scores, episode_lengths = sampler.get_one_epoch_samples(model)

                if episode_scores.shape[0] == 0:
                    continue

                t = timer()
                loss, lr, vf_loss, vf_lr = model.train_one_epoch(
                    observations=observations,
                    actions=actions,
                    rewards=rewards
                )
                dt = 1e7 * (timer() - t) / observations.shape[0]

                model._stochasticity *= .0

                mean_score = episode_scores.mean()
                mean_length = episode_lengths.mean()

                if print_scores:
                    print('[{0}]: loss: {1:.3f}  sc: {2:.3f}  len: {3:.3f}  t: {4:.4f}  lr:{5:.4f}  vfl: {6:.3f}'.format(
                        epoch, loss, mean_score, mean_length, dt, lr, vf_loss
                    ))

                epoch_result = EpochResult(epoch=epoch, loss=loss, score=mean_score)
                experiment_result.epochs.append(epoch_result)

            model.close_sessions()
            if render:
                input()
            return experiment_result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--epoch_episodes', type=int, default=100)
    parser.add_argument('--episode_steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--no-print', action='store_true')
    args = parser.parse_args()

    _debug = args.debug
    if not _debug:
        print('\nUsing simplest formulation of policy gradient.\n')

    result = VanillaPolicyGradientRL(args.env_name, _debug=_debug, unlock_env=True).run_train(
        hidden_layers=(32,),
        epochs=args.epochs,
        epoch_episodes=args.epoch_episodes,
        episode_steps=args.episode_steps,
        learning_rate=args.lr,
        gamma=args.gamma,
        render=args.render,
        print_scores=not args.no_print
    )
