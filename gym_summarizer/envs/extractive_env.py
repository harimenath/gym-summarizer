import gym
import nltk

nltk.download('punkt')
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow_hub as hub

from bert_serving.client import BertClient
from typing import List, Dict


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
        self._init_bert_server(model_dir, max_seq_len)
        self.bc = BertClient()
        self.dim = 768

    def _init_bert_server(self, model_dir, max_seq_len):
        from bert_serving.server.helper import get_args_parser
        from bert_serving.server import BertServer
        args = get_args_parser().parse_args(['-model_dir', model_dir,
                                             '-max_seq_len', max_seq_len,
                                             '-mask_cls_sep',
                                             '-cpu'])
        server = BertServer(args)
        server.start()

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
    def __init__(self, articles_path, summaries_path):
        self.articles: pd.DataFrame = self.load_articles(articles_path)
        self.summaries: Dict = self.load_summaries(summaries_path)

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

        return article, summary


class ExtractiveEnv(gym.Env):
    def __init__(self, data_loader: DataLoader = Duc2007Loader("data/dict-duc-2007-articles.pkl",
                                                               "data/df-duc-2007-gold.pkl"),
                 sent_embedder: SentEmbedder = BertSentEmbedder(),
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
        self.action_space = gym.spaces.Discrete(self.article_len - 1)  # article sentence index selection
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

        return self._get_obs()

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
