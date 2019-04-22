import gym
from gym_summarizer.utils.DataLoader import *
from gym_summarizer.utils.SentenceEmbedder import *


from typing import List, Dict, Tuple
from rouge import Rouge
import random
import warnings
from tqdm import tqdm

SENT_EMBEDDING_DIM = 768


class RewardHelper:
    def __init__(self, reward_name: str, reward_type: str, error_penalty: float = 0, default_reward: float = 0,
                 discount_factor: float = 1.0, is_terminal=False, is_stochastic=False):
        """Helper class for reward shaping in ExtractiveEnv.

        Currently, this helper class allows these reward-shaping use cases:
        1.  Constant-type reward (e.g. all episode rewards are rouge-2 f)
        2.  Scheduled reward (e.g. first 1000 episodes are rouge-1 f, next 1000 episodes are rouge-2 f)
            Requires updating the env's reward helper during training with a stable-baselines callback.
            e.g. env.reward_helper = RewardHelper(**new_params)
        3.  Stochastic reward types.
        4.  Terminal rewards (only returned at end-of-episode) vs intermediate rewards (returned every action).

        :param reward_name: ROUGE algorithm ('rouge-1', 'rouge-2', 'rouge-l')
        :param reward_type: ROUGE type ('f', 'r', 'p') i.e. F1, Precision, Recall
        :param error_penalty: Reward <= 0 to penalize invalid actions (already selected or out-of-range sentences)
        :param default_reward: Reward <=0 to penalize slow episode terminations when is_terminal is True.
        :param is_terminal: Whether to return reward only on episode termination
        :param is_stochastic: Whether to return random ROUGE algorithm/type score
        """
        assert reward_name in ['rouge-1', 'rouge-2', 'rouge-l']
        assert reward_type in ['f', 'r', 'p']

        self.reward_name = reward_name
        self.reward_type = reward_type
        self.error_penalty = error_penalty
        self.default_reward = default_reward
        self.discount_factor = discount_factor
        self.is_terminal = is_terminal
        self.is_stochastic = is_stochastic

        self.reward_calculator = Rouge().get_scores

    def get_reward(self, predicted: str, target: str):
        """Calculates reward (ROUGE) based on specified parameters.

        :param predicted: Agent-generated summary.
        :param target: Gold-standard reference summary.
        :return reward: ROUGE score
        """
        rewards = self.reward_calculator(predicted, target)[0]
        if self.is_stochastic:
            reward = random.choice(sum((list(s.values()) for s in rewards.values()), []))
        else:
            reward = rewards[self.reward_name][self.reward_type]

        return reward


class ExtractiveEnv(gym.Env):
    def __init__(self, data_loader: DataLoader = BatchCNNDMLoader(),
                 reward_helper: RewardHelper = RewardHelper('rouge-2', 'f'),
                 summary_len: int = 4,
                 observation_type: str = None,  # (None, 'cnn_nchw', 'cnn_nhwc', 'maxpool', 'meanpool')
                 verbose: bool = True):
        # helpers
        self.data_loader = data_loader
        self.data_iterator = iter(self.data_loader)  # yields article, tensor, summary
        self.reward_helper = reward_helper
        self.verbose = verbose

        # dimensions
        self.article_len = data_loader.max_len
        self.summary_len = summary_len
        self.sent_embed_dim = data_loader.embed_dim
        self.observation_type = observation_type

        # env attributes
        self.action_space = gym.spaces.Discrete(self.article_len - 1)  # article sentence index selection
        self.observation_space = self._build_observation_space()

        # current episode information
        self.t: int = 0
        self.article: List[str] = None
        self.article_tensor: np.ndarray = np.zeros((self.article_len, self.sent_embed_dim))
        self.summary_pred: str = None  # summary output so far
        self.summary_tensor: np.ndarray = np.zeros((self.summary_len, self.sent_embed_dim))
        self.summary_target: str = None  # gold standard summary
        self.sentences_written = 0
        self.rouge_score = 0
        self.returns = 0
        self.actions: set = set()

        self.reset()

    def _build_observation_space(self):
        if self.observation_type == 'cnn_nhwc':
            observation_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(self.article_len + self.summary_len, self.sent_embed_dim, 1),
                dtype='float32',
            )
        elif self.observation_type == 'cnn_nchw':
            observation_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(1, self.article_len + self.summary_len, self.sent_embed_dim),
                dtype='float32',
            )
        else:
            observation_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(self.article_len + self.summary_len, self.sent_embed_dim),
                dtype='float32',
            )

        return observation_space

    def reset(self):
        """
        - clear everything
        - get new summary

        :return:
        """
        # yield next article/summary
        try:
            self.article, self.article_tensor, self.summary_target = next(self.data_iterator)
        except StopIteration:
            print("Every article has been loaded, restarting data_iterator")
            self.data_iterator = iter(self.data_loader)
            self.article, self.article_tensor, self.summary_target = next(self.data_iterator)

        # reset everything else
        self.t = 0
        self.summary_pred = ""
        self.summary_tensor = np.zeros((self.summary_len, self.sent_embed_dim))
        self.sentences_written = 0
        self.rouge_score = 0
        self.returns = 0
        self.actions = set()

        return self._get_obs()

    def step(self, action: int):
        done = False
        self.t += 1

        if action in self.actions or action >= len(self.article):
            reward = self.reward_helper.error_penalty
            obs = self._get_obs()
            return obs, reward, done, {"summary": self.summary_pred}

        else:
            sentence = self.article[action]
            self.summary_pred += sentence
            self.summary_tensor[self.sentences_written] = self.article_tensor[action]
            self.sentences_written += 1
            self.actions.add(action)
            obs = self._get_obs()

            if self.sentences_written >= min(self.summary_len, len(self.article)):
                done = True
                reward = self._get_reward()
                if self.verbose:
                    print(self.summary_pred, "\n", f"{self.reward_helper.reward_name}-{self.reward_helper.reward_type}",
                          "\n", self.returns, "\n", "-" * 80)

                return obs, reward, done, {"summary": self.summary_pred}

            if self.reward_helper.is_terminal:
                reward = self.reward_helper.default_reward
            else:
                reward = self._get_reward()

        return obs, reward, done, {"summary": self.summary_pred}

    def _get_obs(self):
        obs = np.concatenate([self.article_tensor, self.summary_tensor], axis=0)

        if self.observation_type == 'cnn_nhwc':
            return np.expand_dims(obs, -1)
        elif self.observation_type == 'cnn_nchw':
            return np.expand_dims(obs, 0)
        else:
            return obs

    def _get_reward(self):
        prev_rouge = self.rouge_score
        try:
            self.rouge_score = self.reward_helper.get_reward(target=self.summary_target, predicted=self.summary_pred)
        except ValueError:
            warnings.warn(f"[ROUGE WARNING] Attempting to get ROUGE score for str of len 0.\n"
                          f"summary_pred: {self.summary_pred}\n"
                          f"summary_target: {self.summary_target}\n"
                          f"actions: {self.actions}\n"
                          f"article: {self.article}")
            self.rouge_score = prev_rouge

        reward = self.rouge_score - prev_rouge
        self.returns += reward
        return reward * self.t ** self.reward_helper.discount_factor
