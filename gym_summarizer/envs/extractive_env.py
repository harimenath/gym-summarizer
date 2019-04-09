import gym
import nltk

nltk.download('punkt')
import numpy as np
import pandas as pd
import pickle
import random  # TODO: set seed somewhere
import tensorflow as tf
import tensorflow_hub as hub

from bert_serving.client import BertClient
from typing import List, Dict, NamedTuple, FrozenSet, Tuple

from rouge import Rouge


class SentEmbedder:
    """
    Sentence embeddings with tf-hub Universal Sentence Encoder
    Source: https://tfhub.dev/google/universal-sentence-encoder-large/3
    """

    def __init__(self):
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        self.dim = 512
        self.se_session = tf.Session()
        self.se_session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def embed_article(self, sentences: List[str], max_len: int):
        embeddings = self.se_session.run(self.embed(sentences))
        padded = np.zeros((max_len, self.dim), dtype=np.float32)
        if len(embeddings) > max_len:
            padded[:] = embeddings[:max_len]
        else:
            padded[:len(embeddings)] = embeddings
        return padded


class BertSentEmbedder(SentEmbedder):
    def __init__(self, model_dir="data/bert/uncased_L-12_H-768_A-12/", max_seq_len='NONE'):
        self.bc = self._init_bert_server(model_dir, max_seq_len)
        self.dim = 768

    def _init_bert_server(self, model_dir, max_seq_len):
        try:
            bc = BertClient()
        except:
            from bert_serving.server.helper import get_args_parser
            from bert_serving.server import BertServer
            args = get_args_parser().parse_args(['-model_dir', model_dir,
                                                 '-max_seq_len', max_seq_len,
                                                 '-mask_cls_sep',
                                                 '-cpu',
                                                 '-verbose'])
            server = BertServer(args)
            server.start()
            bc = BertClient()
        return bc
        self.se_session = tf.Session()
        self.se_session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def embed_article(self, sentences: List[str], max_len: int):
        embeddings = self.bc.encode(sentences)
        padded = np.zeros((max_len, self.dim), dtype=np.float32)
        if len(embeddings) > max_len:
            padded[:] = embeddings[:max_len]
        else:
            padded[:len(embeddings)] = embeddings
        return padded


class DataLoader:
    def __init__(self):
        self.articles = ["This is an article. Earlier this week, Bob stated he was not interested in Mary. "
                         "Bob is an amateur carpenter and a vuvuzuela player of world renown. "
                         "Similarly, Mary stated she was not interested in Bob. "
                         "Mary is a professional wrestler and has a farm of axolotls. "
                         "Despite this intrigue, Mary and Bob turned out to both be interested in eachother. "
                         "Furthermore, Bob was also interested in Mary's friend, Steve. "
                         "Steve is on wellfare and mainly mooches off of Mary's wrestling income. "
                         "Reports indicate Mary, Steve and Bob are currently organizing the world's first"
                         "axolotl vuvuzuela-fighting competition. "
                         "Steve's contributions to the effort are unclear as of now. " for _ in range(10)]
        self.summaries = ["Reports indicate Mary, Steve and Bob are currently organizing the world's first"
                          "axolotl vuvuzuela-fighting competition. "
                          "Bob is an amateur carpenter and a vuvuzuela player of world renown. "
                          "Mary is a professional wrestler and has a farm of axolotls. "
                          "Steve is on wellfare and mainly mooches off of Mary's wrestling income. " for _ in range(10)]

    def __iter__(self):
        return zip(self.articles, self.summaries)


class Duc2007Loader(DataLoader):
    def __init__(self, articles_path, summaries_path, sent_embed_len, sent_embedder=BertSentEmbedder()):
        self.articles: pd.DataFrame = self.load_articles(articles_path)
        self.summaries: Dict = self.load_summaries(summaries_path)
        self.sent_embed_len = sent_embed_len
        self.sent_embedder = sent_embedder

    def load_articles(self, articles_path):
        with open(articles_path, 'rb') as f:
            articles = pickle.load(f)
        return articles

    def load_summaries(self, summaries_path):
        df = pd.read_pickle(summaries_path)
        return df

    def __iter__(self):
        self.i = 0
        self.j = 0
        topic = self.summaries.iloc[self.i]['Topic']
        self.topic_articles = self.articles[topic]
        return self

    def __next__(self):
        """Yields article/summary pairs for a given topic.

        Iterates through summaries (i) and through articles with same topic (j).

        :return:
        """
        if self.i >= len(self.summaries):
            raise StopIteration

        # move to next summary when all articles for topic are passed
        if self.j >= len(self.topic_articles):
            self.i += 1
            self.j = 0
            self.topic_articles = self.articles[self.summaries.iloc[self.i]['Topic']]

        article = self.topic_articles[self.j].replace("\n", " ")
        summary = self.summaries.iloc[self.i]['Summary'].replace("\n", " ")

        # tokenize article into sentences and sentence embeddings
        embeddings = self.sent_embedder.embed_article(nltk.tokenize.sent_tokenize(article), self.sent_embed_len)

        self.j += 1

        return article, embeddings, summary


class Duc2007PreEmbedLoader(DataLoader):
    def __init__(self, articles_path, summaries_path, embeddings_path, sent_embed_len):
        self.articles: pd.DataFrame = self.load_articles(articles_path)
        self.summaries: Dict = self.load_summaries(summaries_path)
        self.embeddings: Dict = self.load_summaries(embeddings_path)
        self.sent_embed_len = sent_embed_len

    def load_articles(self, articles_path):
        with open(articles_path, 'rb') as f:
            articles = pickle.load(f)
        return articles

    def load_summaries(self, summaries_path):
        df = pd.read_pickle(summaries_path)
        return df

    def load_precomputed_embeddings(self, embeddings_path):
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings

    def _pad_embeddings(self, embeddings):
        padded = np.zeros((self.sent_embed_len, len(embeddings[0])), dtype=np.float32)
        if len(embeddings) > self.sent_embed_len:
            padded[:] = embeddings[:self.sent_embed_len]
        else:
            padded[:len(embeddings)] = embeddings
        return padded

    def __iter__(self):
        self.i = 0
        self.j = 0
        topic = self.summaries.iloc[self.i]['Topic']
        self.topic_articles = self.articles[topic]
        self.topic_embeddings = self.embeddings[topic]
        return self

    def __next__(self):
        """Yields article/summary pairs for a given topic.

        Iterates through summaries (i) and through articles with same topic (j).

        :return:
        """
        if self.i >= len(self.summaries):
            raise StopIteration

        # move to next summary when all articles for topic are passed
        if self.j >= len(self.topic_articles):
            self.i += 1
            self.j = 0
            self.topic_articles = self.articles[self.summaries.iloc[self.i]['Topic']]
            self.topic_embeddings = self.embeddings[self.summaries.iloc[self.i]['Topic']]

        article = self.topic_articles[self.j].replace("\n", " ")
        embeddings = self._pad_embeddings(self.topic_embeddings[self.j])
        summary = self.summaries.iloc[self.i]['Summary'].replace("\n", " ")

        self.j += 1

        return article, embeddings, summary


class RewardSchedule(NamedTuple):
    reward_name: str = 'rouge-1'  # 'rouge-1', 'rouge-2', 'rouge-l'
    reward_type: str = 'f'  # f, r, p
    start_epoch: int = 0
    # end_epoch: int = None         # # TODO: args for finer-grained control over scheduling. Not implemented yet.
    # start_episode: int = 0
    # end_episode: int = None
    # start_step: int = 0
    # end_step: int = None


# class UnhelpfulRewardHelper:
#     """Helper class for shaping/scheduling reward in summarization env.
#
#     # TODO: revisit this for more complex scheduling if needed. Currently overdesigned af...
#     Note:   This doesn't make sense in the context where scheduling happens across epochs,
#             since we'll be reinstantiating the env each epoch.
#             It's easier to just pass the reward type to the env for each epoch.
#             However, this makes sense for per-episode or per-action scheduling.
#
#     Currently designed for three cases:
#     - Stochastic reward:
#         Randomly select reward for possible rouge scores.
#     - Single reward:
#         Always return same reward type.
#     - Scheduled rewards:
#
#     """
#     raise NotImplementedError
#
#     def __init__(self, schedules: List[RewardSchedule] = None, is_stochastic: bool = False):
#         self.schedule = self._validate_schedules(schedules)
#         self.is_stochastic = is_stochastic
#         self.rouge = Rouge()
#
#         self.current_epoch = 0
#         self.next_switch_epoch = 0
#         self.current_reward_name = None
#         self.current_reward_type = None
#
#     def _validate_schedules(self, schedules: List[RewardSchedule],
#                             allowed_rewards: FrozenSet[str] = frozenset(['rouge-1', 'rouge-2', 'rouge-l']),
#                             allowed_reward_types: FrozenSet[str] = frozenset(['f', 'r', 'p'])
#                             ) -> List[Dict[int, Tuple[str, str]]]:
#         """Sort reward schedules by start epoch and validate values.
#
#         """
#         assert all(s.reward_name in allowed_rewards for s in schedules), \
#             f"Invalid reward in schedule: allowed rewards are: {allowed_rewards}."
#
#         assert all(s.reward_type in allowed_reward_types for s in schedules), \
#             f"Invalid reward type in schedule: allowed rewards are: {allowed_reward_types}."
#
#         assert len(s.start_epoch for s in schedules) == len(set(s.start_epoch for s in schedules)), \
#             "Invalid start epochs in schedule: each reward must have a different start epoch."
#
#         schedule = {s.start_epoch: (s.reward_name, s.reward_type) for s in schedules}
#
#         assert 0 in schedule, "Invalid start epochs in schedule: must include start epoch of 0."
#
#         self.current_reward_name, self.current_reward_type = schedule[0]
#         return schedule
#
#     def rouge_score(self, prediction, target):
#         scores = self.rouge.get_scores(hyps=prediction, refs=target)
#
#         # stochastic case
#         if self.is_stochastic:
#             rewards = sum((list(s.values()) for s in scores.values()), [])  # collapse nested reward dict into list
#             reward = random.choice(rewards)
#
#         # scheduled/constant case
#         else:
#             rouge_name, rouge_type = self._which_rouge_type()
#             reward = scores[rouge_name][rouge_type]
#
#         return reward
#
#     def _which_rouge_type(self):
#         if self.current_epoch in self.schedule:
#             self.current_reward_name, self.current_reward_type = self.schedule[self.current_epoch]
#         return self.current_reward_name, self.current_reward_type


class RewardHelper:
    def __init__(self, reward_name: str, reward_type: str, is_terminal=False, is_stochastic=False):
        """

        :param reward_name: ROUGE algorithm ('rouge-1', 'rouge-2', 'rouge-l')
        :param reward_type: ROUGE type ('f', 'r', 'p') i.e. F1, Precision, Recall
        :param is_terminal: Whether to return reward only on episode termination
        :param is_stochastic: Whether to return random ROUGE algorithm/type score
        """
        assert reward_name in ['rouge-1', 'rouge-2', 'rouge-l']
        assert reward_type in ['f', 'r', 'p']

        self.reward_name = reward_name
        self.reward_type = reward_type
        self.is_terminal = is_terminal
        self.is_stochastic = is_stochastic

        self.reward_calculator = Rouge().get_scores

    def get_reward(self, predicted: str, target: str, terminal: bool):
        """Calculates reward (ROUGE) based on specified parameters.

        :param predicted: Agent-generated summary.
        :param target: Gold-standard reference summary.
        :param terminal: Whether or not state is terminal (end of episode).
        :return reward: ROUGE score
        """
        if self.is_terminal and terminal:
            reward = 0
        else:
            rewards = self.reward_calculator(predicted, target)[0]
            if self.is_stochastic:
                reward = random.choice(sum((list(s.values()) for s in rewards.values()), []))
            else:
                reward = rewards[self.reward_name][self.reward_type]

        return reward


class ExtractiveEnv(gym.Env):
    def __init__(self, data_loader: DataLoader = Duc2007PreEmbedLoader("data/dict-duc-2007-articles.pkl",
                                                                       "data/df-duc-2007-gold.pkl",
                                                                       "data/dict-duc-2007-articles-embed.pkl",
                                                                       250),
                 sent_embedder: SentEmbedder = BertSentEmbedder(),
                 reward_helper: RewardHelper = RewardHelper(reward_name='rouge-2', reward_type='f'),
                 article_dim: int = 50, summary_dim: int = 10,
                 max_summary_len: int = 250):
        # helpers
        self.data_loader = data_loader
        self.data_loader.sent_embed_len = article_dim  # TODO: clean up this hackiness for embedding padding...
        self.data_iterator = iter(self.data_loader)  # yields article(s), summary
        self.sent_embedder = sent_embedder
        self.reward_helper = reward_helper

        # dimensions
        self.article_dim = article_dim
        self.summary_dim = summary_dim
        self.sent_embed_dim = sent_embedder.dim

        # env attributes
        self.action_space = gym.spaces.Discrete(self.article_dim - 1)  # article sentence index selection
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.article_dim + self.summary_dim, self.sent_embed_dim, 1),
            dtype='float32',
        )

        # current episode information
        self.article: List[str] = None
        self.article_tensor: np.ndarray = np.zeros((self.article_dim, self.sent_embed_dim))
        self.summary_pred: str = None  # summary output so far
        self.summary_tensor: np.ndarray = np.zeros((self.summary_dim, self.sent_embed_dim))
        self.summary_target: str = None  # gold standard summary
        self.summary_len = 0
        self.max_summary_len = max_summary_len
        self.sentences_written = 0
        self.rouge_score = 0
        self.actions: set = set()

        self.reset()

    def reset(self):
        # yield next article/summary
        try:
            self.article, self.article_tensor, self.summary_target = next(self.data_iterator)
        except StopIteration:
            print("Every article has been loaded, restarting data_iterator")
            self.data_iterator = iter(self.data_loader)
            self.article, self.article_tensor, self.summary_target = next(self.data_iterator)

        # tokenize article into sentences
        self.article = nltk.tokenize.sent_tokenize(self.article)

        # reset everything else
        self.summary_pred = ""
        self.summary_tensor = np.zeros((self.summary_dim, self.sent_embed_dim))
        self.summary_len = 0
        self.sentences_written = 0
        self.rouge_score = 0
        self.actions = set()

        # reshape tensor (x,y) -> (x,y,1)
        self.article_tensor = self.article_tensor.reshape(self.article_dim, self.sent_embed_dim, 1)
        self.summary_tensor = self.summary_tensor.reshape(self.summary_dim, self.sent_embed_dim, 1)

        return self._get_obs()

    def step(self, action: int):
        done = False

        if action in self.actions or action >= len(self.article):
            reward = -0.1
            obs = self._get_obs()

        else:
            sentence = self.article[action]
            self.summary_pred += sentence
            self.summary_tensor[self.sentences_written] = self.article_tensor[action]
            self.sentences_written += 1
            self.summary_len += len(sentence.split(" "))
            self.actions.add(action)

            if (self.summary_len >= self.max_summary_len) or \
                    (self.sentences_written >= len(self.article)) or \
                    (self.sentences_written >= self.summary_dim):
                self.summary_pred = " ".join(self.summary_pred.split(" ")[:self.max_summary_len])
                done = True
                reward = self._get_reward(done)
                obs = self._get_obs()

                # debugging
                print(self.summary_pred)
                if self.summary_len < self.max_summary_len:
                    print(self.summary_len)

                self.reset()

                return obs, reward, done, {}

            reward = self._get_reward(done)
            obs = self._get_obs()

        return obs, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.article_tensor, self.summary_tensor], axis=0)

    def _get_reward(self, done):
        prev_rouge = self.rouge_score
        self.rouge_score = self.reward_helper.get_reward(target=self.summary_target, predicted=self.summary_pred,
                                                         terminal=done)
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
