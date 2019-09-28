from timeit import default_timer as timer
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model

from spinup.utils.clr import cyclic_learning_rate as clr
from spinup.utils.experiment import ExperimentResult, ExperimentInfo, EpochResult
from spinup.utils.my_utils import managed_gym_environment


class VanillaPolicyGradientRL:
    class _Model:
        def __init__(
                self, observation_shape: Tuple[int, ...], n_actions: int,
                hidden_layers: Tuple[int, ...],
                learning_rate: float = 1e-2,
                gamma: float = .99,
                stochasticity: float = .1
        ):
            self._n_actions = n_actions
            self._stochasticity = stochasticity

            # reset default graph
            tf.reset_default_graph()

            # define inputs
            self._state = tf.placeholder(shape=(None,) + observation_shape, dtype=tf.float32)
            self._action = tf.placeholder(shape=(None,), dtype=tf.int32)
            self._reward = tf.placeholder(shape=(None,), dtype=tf.float32)
            self._total_reward = tf.placeholder(shape=(None,), dtype=tf.float32)
            self._n_state = tf.placeholder(shape=(None,) + observation_shape, dtype=tf.float32)

            # shared hidden part of NN
            input_layer, hidden_nn = self._get_hidden_layers(observation_shape, hidden_layers)
            input_layer2, hidden_nn2 = self._get_hidden_layers(observation_shape, hidden_layers)

            # Policy(s) and V(s) approximates
            policy = self._approximate_policy(input_layer, hidden_nn, n_actions)
            V = self._approximate_value(input_layer, hidden_nn)

            # Value prediction
            s, a, r, ns, tot_r = self._state, self._action, self._reward, self._n_state, self._total_reward
            pred_n_v = V(ns)
            pred_v = V(s)
            Qsa = r + gamma * pred_n_v

            # Policy sampling for action prediction
            logits = policy(s)
            self._pred_action = self._sample_action(logits)

            # Policy loss
            A = tot_r - pred_v
            self._policy_loss = self._get_policy_loss(n_actions, A, self._action, logits)

            # Value-function loss
            self._value_func_loss = self._get_value_func_loss(pred_v, Qsa)

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
        def _approximate_policy(input_layer, hidden_nn, n_actions: int):
            output_layer = Dense(n_actions)(hidden_nn)
            return Model(input_layer, output_layer)

        @staticmethod
        def _approximate_value(input_layer, hidden_nn):
            output_layer = tf.squeeze(Dense(1)(hidden_nn), axis=1)
            return Model(input_layer, output_layer)

        @staticmethod
        def _sample_action(logits):
            return tf.squeeze(tf.random.categorical(logits=logits, num_samples=1), axis=1)

        @staticmethod
        def _get_value_func_loss(pred_v, target_Qsa):
            return tf.losses.mean_squared_error(pred_v, tf.stop_gradient(target_Qsa))

        @staticmethod
        def _get_policy_loss(
                n_actions: int, advantage: tf.Tensor, action: tf.Variable, logits: tf.Tensor
        ):
            action_one_hot = tf.one_hot(action, n_actions)
            log_probs = tf.nn.log_softmax(logits)
            log_prob = tf.reduce_sum(action_one_hot * log_probs, axis=1)
            loss = -tf.reduce_mean(log_prob * tf.stop_gradient(advantage))
            return loss

        @staticmethod
        def _get_hidden_layers(observation_shape: Tuple[int, ...], hidden_layers: Tuple[int, ...]):
            input_layer = Input(shape=observation_shape, dtype=tf.float32)
            x = input_layer
            for hidden_layer in hidden_layers:
                x = Dense(hidden_layer, activation=tf.nn.relu)(x)
            return input_layer, x

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

        def train_one_epoch(self, states, actions, rewards, n_states, total_rewards):
            value_func_loss, _, value_func_learning_rate = self._session.run(
                [self._value_func_loss, self._value_func_train_op, self._value_func_learning_rate],
                feed_dict={
                    self._state: states,
                    self._reward: rewards,
                    self._n_state: n_states,
                    self._total_reward: total_rewards
                }
            )
            policy_loss, _, policy_learning_rate = self._session.run(
                [self._policy_loss, self._policy_train_op, self._policy_learning_rate,],
                feed_dict={
                    self._state: states,
                    self._action: actions,
                    self._reward: rewards,
                    self._n_state: n_states,
                    self._total_reward: total_rewards
                }
            )
            return policy_loss, policy_learning_rate, value_func_loss, value_func_learning_rate

        def get_loss(self, observations, actions, rewards):
            loss = self._session.run(
                self._policy_loss,
                feed_dict={
                    self._state: observations,
                    self._action: actions,
                    self._reward: rewards,
                }
            )
            return loss

        def predict_action(self, observations):
            actions_batch = self._session.run(self._pred_action, {
                self._state: observations.reshape(1, -1)
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
            self._total_rewards = np.empty(shape=(max_total_steps,), dtype=np.float32)
            self._current_rewards = np.empty(shape=(max_total_steps,), dtype=np.float32)
            self._episode_scores = np.empty(shape=(n_episodes,), dtype=np.float32)
            self._episode_lengths = np.empty(shape=(n_episodes,), dtype=np.int32)

            observation = self._env.reset()
            observations_shape = (max_total_steps,) + observation.shape
            self._states = np.empty(shape=observations_shape, dtype=np.float32)
            self._n_states = np.empty_like(self._states)

        def get_one_epoch_samples(self, model: 'VanillaPolicyGradientRL._Model'):
            step, episode, done, ep_step = 0, 0, False, 0
            state = self._env.reset()

            while episode < self._n_episodes:
                if episode == 0 and self._render:
                    self._env.render()

                self._states[step] = state.copy()

                action = model.predict_action(state)
                self._actions[step] = action

                state, reward, done, _ = self._env.step(action)
                self._current_rewards[ep_step] = reward
                self._rewards[step] = reward

                ep_step += 1
                step += 1
                if done or ep_step >= self._episode_steps:
                    score = self._current_rewards[:ep_step].sum()
                    self._total_rewards[(step - ep_step):step] = self._reward_to_go(
                        self._current_rewards[:ep_step]
                    )
                    self._episode_scores[episode] = score
                    self._episode_lengths[episode] = ep_step
                    self._n_states[(step - ep_step):(step - 1)] = self._states[(step - ep_step + 1):step]
                    self._n_states[step - 1] = state

                    episode += 1
                    ep_step = 0
                    state = self._env.reset()

            states = self._states[:step]
            actions = self._actions[:step]
            rewards = self._rewards[:step]
            total_rewards = self._total_rewards[:step]
            n_states = self._n_states[:step]
            episode_scores = self._episode_scores[:episode]
            episode_lengths = self._episode_lengths[:episode]

            return states, actions, rewards, n_states, total_rewards, episode_scores, episode_lengths

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
                gamma=gamma,
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
                states, actions, rewards, n_states, total_rewards, episode_scores, episode_lengths = \
                        sampler.get_one_epoch_samples(model)

                if episode_scores.shape[0] == 0:
                    continue

                t = timer()
                loss, lr, vf_loss, vf_lr = model.train_one_epoch(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    n_states=n_states,
                    total_rewards=total_rewards
                )
                dt = 1e7 * (timer() - t) / states.shape[0]

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
        hidden_layers=(32, ),
        epochs=args.epochs,
        epoch_episodes=args.epoch_episodes,
        episode_steps=args.episode_steps,
        learning_rate=args.lr,
        gamma=args.gamma,
        render=args.render,
        print_scores=not args.no_print
    )
