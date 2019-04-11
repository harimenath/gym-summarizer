import gym
import nltk

nltk.download('punkt')
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.core.example import example_pb2
import tensorflow_hub as hub
from pathlib import Path
import struct

from bert_serving.client import BertClient
from typing import List, Dict, Tuple
from rouge import Rouge
import random
import warnings
from tqdm import tqdm

SENT_EMBEDDING_DIM = 768


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
        assert Path(model_dir).exists()
        self.bc = self._init_bert_client(model_dir, max_seq_len)
        self.dim = SENT_EMBEDDING_DIM

    def _init_bert_client(self, model_dir, max_seq_len):
        """Initialize bert client for sentence embeddings and avoid restarting bert-server if already running.

        Bert-server can take a long time to start, pollute stdout during training, and create many temp log files.
        It's highly recommended to run bert-server beforehand from command-line.

        :param model_dir: directory containing bert model
        :param max_seq_len: max sequence length for bert
        :return bc: bert-client
        """
        try:
            bc = BertClient()
        except:
            from bert_serving.server.helper import get_args_parser
            from bert_serving.server import BertServer
            args = get_args_parser().parse_args(['-model_dir', model_dir,
                                                 '-max_seq_len', max_seq_len,
                                                 '-mask_cls_sep',
                                                 '-cpu'])
            server = BertServer(args)
            server.start()
            bc = BertClient()

        return bc

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


class DummySentEmbedder(SentEmbedder):
    """
    For testing purposes
    """

    def __init__(self):
        self.dim = 768

    def embed_article(self, sentences: List[str], max_len: int):
        return np.random.randn(max_len, self.dim)


class CNNDMLoader(DataLoader):
    def __init__(self, finished_files_path: str = 'data/finished_files/', sent_embedder: SentEmbedder = None,
                 max_number_sentences: int = 20,
                 sent_embedding_dim=SENT_EMBEDDING_DIM):
        """
        This DataLoader loads the "finished_files" from tokenized CNN-DM dataset from https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail
        We keep track of sentence embeddings for each sentence of each article as well as the full articles and summaries.

        Parameters
        ---------
        finished_files_path: ``str``
            Path to the  finished_files directory.

        sent_embedder: ``SentEmbedder``
            Object that embeds sentences into a fixed vector space.
        """

        assert Path(finished_files_path).exists()

        self.sent_embedder = sent_embedder
        self.sent_embedding_dim = sent_embedding_dim
        self.max_number_sentences = max_number_sentences

        # Training
        self.train_articles, self.train_summaries = self.load_articles_and_summaries(
            Path(finished_files_path) / 'train.bin')
        print("Loading train embeddings...")
        self.train_article_tensors = self.load_embeddings(Path(finished_files_path) / 'train.vectorized.npz')

        # # Validation
        # self.validation_articles, self.validation_summaries = self.load_articles_and_summaries(Path(finished_files_path) / 'val.bin')
        # print("Loading validation embeddings...")
        # self.val_article_tensors = self.load_embeddings(Path(finished_files_path) / 'val.vectorized.npz')

        # # Testing
        # self.test_articles, self.test_summaries = self.load_articles_and_summaries(Path(finished_files_path) / 'test.bin')
        # print("Loading test embeddings...")
        # self.test_article_tensors = self.load_embeddings(Path(finished_files_path) / 'test.vectorized.npz')

    def load_articles_and_summaries(self, path_to_binary: Path) -> Tuple[List[List[str]], List[str]]:
        """
        Loads articles and summaries from binary files as described in CNNDMLoader docstring.

        Parameters
        ----------

        path_to_binary: ``str``
            Path to a train.bin, valid.bin, etc.

        Returns
        -------
        Tuple[List[List[str]], List[str]]
            A list of: string articles split by sentence, and summaries respectively.
        """

        print(f"Loading articles and summaries from path: {path_to_binary}...")
        assert path_to_binary.exists()

        articles: List[List[str]] = []
        summaries: List[str] = []

        reader = open(path_to_binary, 'rb')
        while True:
            len_bytes = reader.read(8)
            if not len_bytes: break  # finished reading this file
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            example = example_pb2.Example.FromString(example_str)
            try:
                article = example.features.feature['article'].bytes_list.value[0].decode('utf-8')
                summary = example.features.feature['abstract'].bytes_list.value[0].decode('utf-8')
                if len(article) != 0:
                    articles.append(nltk.sent_tokenize(article))
                    summaries.append(summary.replace("<s>", "").replace("</s>", ""))
            except ValueError:
                print("Failed retrieving an article or abstract.")

        print(f"Articles and summaries read from path: {path_to_binary}.")

        return articles, summaries

    def load_embeddings(self, embedding_path: Path) -> np.ndarray:
        assert embedding_path.exists()
        print(f"Loading embeddings from path: {embedding_path}...")
        archive = np.load(embedding_path)

        return archive['arr_0']

    def embed_articles(self, articles: List[List[str]] = [], outfile: str = '') -> None:
        """
        Embeds and saves articles to a .npz file.
        """
        assert self.sent_embedder is not None
        print(f"Saving to path: {outfile}.")

        # TODO: avoid padding when precomputing/embedding articles
        articles_tensor: np.ndarray = np.zeros((len(articles), self.max_number_sentences, self.sent_embedder.dim))

        # TODO: something fucky is going on here........look into it
        for article_idx, article in tqdm(enumerate(articles), total=len(articles)):
            articles_tensor[article_idx] = self.sent_embedder.embed_article(article_sentences,
                                                                            self.max_number_sentences)

        np.savez_compressed(outfile, articles_tensor)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self) -> Tuple[List[str], np.ndarray, str]:

        if self.i == len(self.train_articles):
            raise StopIteration
        else:
            article = self.train_articles[self.i]
            article_embedding = self.train_article_tensors[self.i]
            summary = self.train_summaries[self.i]

            self.i += 1

            return article, article_embedding, summary


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
        self.j += 1

        return article, summary


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

    def get_reward(self, predicted: str, target: str, t: int):
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

        return reward * self.discount_factor ** t


class ExtractiveEnv(gym.Env):
    def __init__(self, data_loader: DataLoader = CNNDMLoader("data/finished_files/subset/"),
                 reward_helper: RewardHelper = RewardHelper('rouge-2', 'f'),
                 article_len: int = 20, summary_len: int = 4,
                 observation_type: str = None,  # (None, 'cnn_nchw', 'cnn_nhwc', 'maxpool', 'meanpool')
                 verbose: bool = True):
        # helpers
        self.data_loader = data_loader
        self.data_iterator = iter(self.data_loader)  # yields article(s), summary
        self.reward_helper = reward_helper
        self.verbose = verbose

        # dimensions
        self.article_len = article_len
        self.summary_len = summary_len
        self.sent_embed_dim = data_loader.sent_embedding_dim
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
                if self.verbose: print(self.summary_pred, "\n", "-" * 80)

            if not done and self.reward_helper.is_terminal:
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
            self.rouge_score = self.reward_helper.get_reward(target=self.summary_target, predicted=self.summary_pred,
                                                             t=self.t)
        except ValueError:
            warnings.warn(f"[ROUGE WARNING] Attempting to get ROUGE score for str of len 0.\n"
                          f"summary_pred: {self.summary_pred}\n"
                          f"summary_target: {self.summary_target}\n"
                          f"actions: {self.actions}\n"
                          f"article: {self.article}")
            self.rouge_score = prev_rouge

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


if __name__ == "__main__":
    dataloader = CNNDMLoader('data/finished_files/subset/')
    dataloader = CNNDMLoader(finished_files_path='data/finished_files/subset/', sent_embedder=BertSentEmbedder())
    dataloader.embed_articles(dataloader.train_articles, outfile='data/finished_files/subset/train.vectorized.npz')
