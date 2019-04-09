from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpLstmPolicy, FeedForwardPolicy, LstmPolicy, CnnPolicy
from gym.envs.registration import register
from stable_baselines import DQN, A2C
import gym
import os
import numpy as np
from gym_summarizer.envs.extractive_env import RewardHelper

best_mean_reward, n_steps = -np.inf, 0


def callback(_locals, _globals):
    # TODO: modify env RewardHelper to change reward between epochs (avoid recreating env for plotting reasons)
    global n_steps, best_mean_reward
    if (n_steps + 1) % 100 == 0:
        print(n_steps)
    n_steps += 1
    return True


register(
    id='ExtractiveEnv-v0',
    entry_point='gym_summarizer.envs:ExtractiveEnv',
)

# Create log dir
log_dir = "logs/gym"
os.makedirs(log_dir, exist_ok=True)
env_name = 'ExtractiveEnv-v0'


experiment_name = "A2C_rouge2_intermediate_0.0.2"
env = gym.make(env_name, reward_helper=RewardHelper(reward_name='rouge-2', reward_type='f', is_terminal=False))
env = DummyVecEnv([lambda: env])
# model = A2C(Policy, env,ent_coef=0.1, verbose=1)
model = A2C(LstmPolicy, env, tensorboard_log=f"../logs/tensorboard/{experiment_name}/",ent_coef=0.1, verbose=1)
model.learn(total_timesteps=100000, callback=callback)
model.save(experiment_name)

experiment_name = "A2C_rouge2_terminal_0.0.1"
env = gym.make(env_name, reward_helper=RewardHelper(reward_name='rouge-2', reward_type='f', is_terminal=True))
env = DummyVecEnv([lambda: env])
# model = A2C(Policy, env, ent_coef=0.1, verbose=1)
model = A2C(MlpLstmPolicy, env, tensorboard_log=f"../logs/tensorboard/{experiment_name}/",ent_coef=0.1, verbose=1)
model.learn(total_timesteps=100000, callback=callback)
model.save(experiment_name)

"""
0.0.0
    Episode reward for terminal reward was always -0.4 (i.e. agent never selected sentences)
    However, advantage went up to 0, indicating all actions equally bad...?
        
    Addendum: upon further inspection, the issue seems to have calling _get_obs() before updating state.
    We address this in 0.0.1; 
        - call _get_obs after state update
        - no longer count invalid actions towards "sentences_written".
        - enforce summary length of min(max_summary_len, article_len)
    
0.0.1

0.0.2
    LSTM without MLP
    
"""