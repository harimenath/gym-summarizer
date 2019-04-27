import pickle
import plac
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
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
            if n_episodes % 100 == 0:
                print(n_episodes)

    return eval_rewards, eval_summaries, eval_actions



@plac.annotations(
    model_path=plac.Annotation("Path to model", type=str),
)
def main(model_path):
    # load model
    class LeadBaselineModel():
        def __init__(self, summary_len: int = 4):
            self.summary_len = summary_len

        def predict(self, obs):
            summary = obs[-self.summary_len:]
            for i in range(len(obs)):
                if sum(summary[min(i, self.summary_len - 1)]) == 0:
                    return i, 0

    if model_path == 'experiments/baseline':
        model = LeadBaselineModel()
    else:
        model = A2C.load(model_path)

    # define eval_env
    register(
        id='ExtractiveEnv-v0',
        entry_point='gym_summarizer.envs:ExtractiveEnv',
    )

    eval_dataloader = BatchCNNDMLoader('data/finished_files/test/')
    eval_env = gym.make('ExtractiveEnv-v0', data_loader=eval_dataloader,
                        reward_helper=RewardHelper('average', 'f', is_terminal=True),
                        verbose=False)
    eval_env.summary_len_override = 3
    eval_rewards, eval_summaries, eval_actions = evaluate(model, eval_env)
    with open(f"{model_path}_eval_rewards.pkl", 'wb') as f:
        pickle.dump(eval_rewards, f)

    with open(f"{model_path}_eval_summaries.pkl", 'wb') as f:
        pickle.dump(eval_summaries, f)

    with open(f"{model_path}_eval_actions.pkl", 'wb') as f:
        pickle.dump(eval_actions, f)

    if model_path == "experiments/cnndm_0.3.0_a2c-mlp_intermediate_rl.model":
        experiment = "Constant Reward (Intermediate)"
    if model_path == "experiments/cnndm_0.3.1_a2c-mlp_intermediate_r1-2-l.model":
        experiment = "Scheduled Reward (Intermediate)"
    if model_path == "experiments/cnndm_0.3.2_a2c-mlp_terminal_rl.model":
        experiment = "Constant Reward (Terminal)"
    if model_path == "experiments/cnndm_0.3.3_a2c-mlp_terminal_r1-2-l.model":
        experiment = "Scheduled Reward (Terminal)"
    if model_path == "experiments/baseline":
        experiment = "Lead 3 Baseline"

    # plot actions histogram
    df_actions = pd.DataFrame(sum(eval_actions, []))
    actions = df_actions[0].value_counts()
    action_space = eval_env.action_space.n
    for a in range(action_space):
        if a not in actions:
            actions[a] = 0
    first_sentence = actions[0]
    actions[0] = 0
    plt.bar(actions.index, actions.values, align='center')
    plt.xticks(range(0, action_space, action_space//10))
    plt.xlabel('Action')
    plt.ylabel('Action Count')
    plt.title(f"Evaluation: {experiment} \nFirst sentence count: {first_sentence}")
    plt.show()

    # plot rewards
    df_rewards = pd.DataFrame(eval_rewards)
    plt.plot(df_rewards[0], 'lightblue', df_rewards[0].rolling(100).mean(), 'b')
    plt.xlabel('Episodes')
    plt.ylabel('Average ROUGE-1/2/L')
    plt.title(f"Evaluation: {experiment} \nmean_score={np.mean(eval_rewards).round(4)}")
    plt.show()

    print(eval_summaries[0])


if __name__ == '__main__':
    plac.call(main)