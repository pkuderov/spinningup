from typing import List, Tuple
from dataclasses import asdict
from os import path
from tqdm import trange, tqdm

from spinup.utils.experiment import ExperimentResult


def get_runner_po_value_func_baseline(arguments) -> ExperimentResult:
    from spinup.my_algos.simple_pg.pg_03_baselines_value_func import VanillaPolicyGradientRL

    return VanillaPolicyGradientRL(arguments.env_name, _debug=arguments.debug, unlock_env=True).run_train(
        hidden_layers=(32,),
        epochs=arguments.epochs,
        epoch_episodes=arguments.epoch_episodes,
        episode_steps=arguments.episode_steps,
        learning_rate=arguments.lr,
        gamma=arguments.gamma,
        render=arguments.render,
        print_scores=not arguments.no_print
    )


def get_runner_actor_critic(arguments) -> ExperimentResult:
    from spinup.my_algos.simple_pg.pg_05_advantage_func import VanillaPolicyGradientRL

    return VanillaPolicyGradientRL(arguments.env_name, _debug=arguments.debug, unlock_env=True).run_train(
        hidden_layers=(32,),
        epochs=arguments.epochs,
        epoch_episodes=arguments.epoch_episodes,
        episode_steps=arguments.episode_steps,
        learning_rate=arguments.lr,
        gamma=arguments.gamma,
        render=arguments.render,
        print_scores=not arguments.no_print
    )

def get_runner_actor_critic_non_shared(arguments) -> ExperimentResult:
    from spinup.my_algos.simple_pg.pg_05_advantage_func_non_shared import VanillaPolicyGradientRL

    return VanillaPolicyGradientRL(arguments.env_name, _debug=arguments.debug, unlock_env=True).run_train(
        hidden_layers=(32,),
        epochs=arguments.epochs,
        epoch_episodes=arguments.epoch_episodes,
        episode_steps=arguments.episode_steps,
        learning_rate=arguments.lr,
        gamma=arguments.gamma,
        render=arguments.render,
        print_scores=not arguments.no_print
    )


def save_results(filename: str, results: List[Tuple[str, ExperimentResult]]):
    import pandas as pd
    df = pd.DataFrame([
        dict({'experiment': f'{name}_{res.info.to_short_str()}' }, **asdict(epoch_res))
        for name, res in results
        for epoch_res in res.epochs
    ])
    print('results saved to', path.abspath(filename))
    df.to_csv(filename, sep=';', index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--experiments', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epoch_episodes', type=int, default=10)
    parser.add_argument('--episode_steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--no_print', action='store_true', default=True)
    parser.add_argument('--save_to', type=str, default='./out/last.csv')

    args = parser.parse_args()

    runners = [
        ('baseline', get_runner_po_value_func_baseline),
        ('actor-critic', get_runner_actor_critic),
        ('actor-critic-non-shared', get_runner_actor_critic_non_shared),
    ]

    experiments: int = args.experiments
    results = []
    for runner_name, runner in tqdm(runners, desc='runners'):
        for experiment in trange(experiments, desc='experiments'):
            result = runner(args)
            results.append((runner_name, result))

    save_results(args.save_to, results)
