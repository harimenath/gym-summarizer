from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy
from stable_baselines.deepq.policies import LnMlpPolicy
from policies import alternative_cnn
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds

from gym_summarizer.envs.extractive_env import RewardHelper

from gym.envs.registration import register
from stable_baselines import DQN, A2C, PPO2
import gym
import os
import numpy as np

best_mean_reward, n_steps = -np.inf, 0


def callback(_locals, _globals):
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

# create env(s)
multiproc = False
env_name = 'ExtractiveEnv-v0'


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id, observation_type='cnn_nhwc')
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


# experiment_name = "cnndm_0.0.2"
# env = gym.make(env_name)
# env = DummyVecEnv([lambda: env])
# model = A2C(MlpLstmPolicy, env, tensorboard_log=f"../logs/tensorboard/{experiment_name}/",ent_coef=0.1, verbose=1)

# experiment_name = "cnndm_0.0.3"
# env = SubprocVecEnv([make_env(env_name, i) for i in range(4)])
# model = PPO2("CnnLstmPolicy", env, tensorboard_log=f"../logs/tensorboard/{experiment_name}/",ent_coef=0.1, verbose=1,
#              policy_kwargs={"cnn_extractor": alternative_cnn})


# experiment_name = "cnndm_0.0.4"
# env = gym.make(env_name, observation_type='cnn_nhwc')
# env = DummyVecEnv([lambda: env])
# model = PPO2(MlpPolicy, env, tensorboard_log=f"../logs/tensorboard/{experiment_name}/",ent_coef=0.1, verbose=1)

# experiment_name = "cnndm_0.0.5"
# env = gym.make(env_name, observation_type='cnn_nhwc')
# env = DummyVecEnv([lambda: env])
# model = DQN(LnMlpPolicy, env, tensorboard_log=f"../logs/tensorboard/{experiment_name}/", verbose=1)

experiment_name = "cnndm_0.0.6"
env = gym.make(env_name, reward_helper=RewardHelper(reward_name='rouge-2', reward_type='f',
                                                    is_terminal=True, default_reward=-0.01, error_penalty=-0.1))
env = DummyVecEnv([lambda: env])
model = A2C(MlpPolicy, env, tensorboard_log=f"../logs/tensorboard/{experiment_name}/", verbose=1)
model.learn(total_timesteps=100000, callback=callback)
