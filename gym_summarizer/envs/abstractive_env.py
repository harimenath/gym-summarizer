import gym
import nltk
from typing import List, Dict
import numpy as np


class AbstractiveEnv(gym.Env):
    def __init__(self, data_loader: DataLoader = Duc2007Loader("data/dict-duc-2007-articles.pkl",
                                                               "data/df-duc-2007-gold.pkl"),
                 sent_embedder: SentEmbedder = SentEmbedder(),
                 article_len: int = 20, summary_len: int = 4):
        # helpers
        self.data_loader = data_loader
        self.data_iterator = iter(self.data_loader)  # yields article(s), summary
        self.sent_embedder = sent_embedder

        # dimensions
        self.article_len = article_len
        self.summary_len = summary_len
        self.sent_embed_dim = sent_embedder.dim

        # env attributes
        self.action_space = gym.spaces.Discrete(self.article_len-1)  # article sentence index selection
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.article_len + self.summary_len, self.sent_embed_dim),
            dtype='float32',
        )

        # current episode information
        self.article: List[str] = None
        self.article_tensor: np.ndarray = np.zeros((self.article_len, self.sent_embed_dim))
        self.summary_pred: str = None  # summary output so far
        self.summary_tensor: np.ndarray = np.zeros((self.summary_len, self.sent_embed_dim))
        self.summary_target: str = None  # gold standard summary
        self.sentences_written = 0
        self.rouge_score = 0
        self.actions: set = set()

        self.reset()

    def reset(self):
        """
        - clear everything
        - get new summary

        :return:
        """
        # yield next article/summary
        try:
            self.article, self.summary_target = next(self.data_iterator)
        except StopIteration:
            print("Every article has been loaded, restarting data_iterator")
            self.article, self.summary_target = next(self.data_iterator)

        # tokenize article into sentences and update article_tensor with sentence embeddings
        self.article = nltk.tokenize.sent_tokenize(self.article)
        self.article_tensor = self.sent_embedder.embed_article(self.article, self.article_len)

        # reset everything else
        self.summary_pred = ""
        self.summary_tensor = np.zeros((self.summary_len, self.sent_embed_dim))
        self.sentences_written = 0
        self.rouge_score = 0
        self.actions = set()

    def step(self, action: int):
        done = False

        if action in self.actions or action >= len(self.article):
            reward = -0.1

        else:
            self.summary_pred += self.article[action]
            self.summary_tensor[self.sentences_written] = self.article_tensor[action]
            self.sentences_written += 1
            self.actions.add(action)
            reward = self._get_reward()

        obs = self._get_obs()

        if self.sentences_written >= self.summary_len:
            done = True
        return obs, reward, done, {"summary": self.summary_pred}

    def _get_obs(self):
        return np.concatenate([self.article_tensor, self.summary_tensor], axis=0)

    def _get_reward(self):
        prev_rouge = self.rouge_score
        self.rouge_score = self._lcs_rouge(target=self.summary_target, predicted=self.summary_pred)
        return self.rouge_score - prev_rouge

    def _lcs_rouge(self, target, predicted):
        """ROUGE-L implementation adapted from: https://github.com/robertnishihara/ray-tutorial-docker

        """
        string = target
        sub = predicted

        if len(string) < len(sub):
            sub, string = string, sub

        lengths = [[0 for _ in range(0, len(sub) + 1)] for _ in range(0, len(string) + 1)]

        for j in range(1, len(sub) + 1):
            for i in range(1, len(string) + 1):
                if string[i - 1] == sub[j - 1]:
                    lengths[i][j] = lengths[i - 1][j - 1] + 1
                else:
                    lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

        lcs_len = lengths[len(string)][len(sub)]

        if len(predicted) == 0 or len(target) == 0:
            return 0.0

        prec_max = lcs_len / float(len(predicted))
        rec_max = lcs_len / float(len(target))
        beta = 1.2

        if prec_max != 0 and rec_max != 0:
            score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
        else:
            score = 0.0

        return score
