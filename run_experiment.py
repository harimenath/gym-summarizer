import pickle
import plac
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

import gym
from gym.envs.registration import register

from gym_summarizer.envs.extractive_env import RewardHelper
from gym_summarizer.utils.DataLoader import BatchCNNDMLoader


# define evaluate function
def evaluate(model, eval_env, early_stop = None):
    """
    Use terminal reward env to only get summary rouge score.
    :param model:
    :param eval_env:
    :return:
    """
    obs = eval_env.reset()
    done = False

    eval_rewards = []
    eval_summaries = []
    eval_actions = []
    episode_actions = []
    n_episodes = 0

    while True:
        # exit after going through test set once
        if eval_env.num_epochs > 0 or (early_stop is not None and n_episodes >= early_stop):
            break

        action, _states = model.predict(obs)
        obs, reward, done, info = eval_env.step(action)

        episode_actions.append(action)
        if done:
            eval_rewards.append(reward)
            eval_actions.append(episode_actions)
            eval_summaries.append(info['summary'])
            episode_actions = []
            obs = eval_env.reset()
            n_episodes += 1
            print(n_episodes)

    return eval_rewards, eval_summaries, eval_actions


n_episodes = 1
returns = []
eval_scores = []
eval_increment = None
rh_schedule = None
rh_kwargs = None
model = None
eval_env = None


def reward_schedule_callback(_locals, _globals):
    global n_episodes, rh_schedule, returns, eval_rewards
    if n_episodes == _locals["self"].env.envs[0].num_episodes:
        n_episodes += 1
        returns.append(_locals["self"].env.envs[0].prev_returns)

        if n_episodes % 100 == 0:
            print(f"episode: {n_episodes}")

        if n_episodes % eval_increment == 0:
            eval_rewards, eval_summaries, eval_actions = evaluate(model, eval_env, early_stop=10)
            print(eval_rewards, eval_summaries, eval_actions)
            eval_scores.append(np.mean(eval_rewards))

        if n_episodes in rh_schedule:
            _locals["self"].env.envs[0].reward_helper = RewardHelper(**rh_schedule[n_episodes], **rh_kwargs)
            print("-" * 80, "\n", f"SWITCHED TO {rh_schedule[n_episodes]['reward_name']}", "\n", "-" * 80)

    return True

@plac.annotations(
    experiment_params=plac.Annotation("Path to .json containing experiment params", type=str),
)
def main(experiment_params):


    # define experiment params
    with open(experiment_params, "rb") as f:
       params = json.load(f)

    global eval_increment, rh_kwargs, rh_schedule, model, eval_env # global variables for callback fn
    experiment = params['experiment_name']
    env_name = params['env_name']
    rh_kwargs = params['rh_kwargs']
    rh_schedule = {int(k): v for k,v in params['rh_schedule'].items()}
    verbose_env = params['verbose_env']
    total_steps = params['total_steps']
    train_subset = params['train_subset']
    eval_increment = params['eval_increment']


    # setup env
    register(
        id=env_name,
        entry_point=f'gym_summarizer.envs:{env_name.split("-")[0]}',
    )

    reward_helper = RewardHelper(**rh_schedule[0], **rh_kwargs)

    env = gym.make(env_name, reward_helper=reward_helper, verbose=verbose_env)
    env.data_loader.early_stop = train_subset
    env = DummyVecEnv([lambda: env])

    # define callback function and evaluation env
    eval_dataloader = BatchCNNDMLoader('data/finished_files/test/')
    eval_env = gym.make(env_name, data_loader=eval_dataloader,
                        reward_helper=RewardHelper('average', 'f', is_terminal=True),
                        verbose=False)



    # define model and learn
    model = A2C(MlpPolicy, env, tensorboard_log=f"experiments/{experiment}/", verbose=0, n_steps=2)
    model.learn(total_timesteps=total_steps, callback=reward_schedule_callback)

    # save model and callbacks output
    model.save(f"{experiment}.model")
    with open(f"experiments/{experiment}_returns.pkl", 'wb') as f:
        pickle.dump(returns, f)

    with open(f"experiments/{experiment}_eval.pkl", 'wb') as f:
        pickle.dump(eval_scores, f)

    # plot returns
    df = pd.DataFrame(returns)
    plt.plot(df[0], 'lightblue', df[0].rolling(1000).mean(), 'blue')
    plt.title(f'Training: {experiment}')
    plt.xlabel('Num Episodes')
    plt.ylabel('Episode Reward')
    plt.legend(['Raw', 'Smoothed'])
    plt.show()


if __name__ == '__main__':
    plac.call(main)