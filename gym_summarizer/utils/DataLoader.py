from pathlib import Path
import pickle
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import warnings

from gym_summarizer.utils.SentenceEmbedder import SentenceEmbedder, DummySentenceEmbedder, BertSentenceEmbedder



class DataLoader(object):
    """Iterator class for loading data into environment.

    """
    def __init__(self):
        self.article_len: int = 100
        self.embed_dim: int = 768

    def __iter__(self) -> 'DataLoader':
        raise NotImplementedError

    def __next__(self) -> Tuple[List[str], np.ndarray, str]:
        raise NotImplementedError


class DummyDataLoader(DataLoader):
    """Dummy data loader for testing purposes.

    """
    def __init__(self, max_len: int = 10, embed_dim: int = 10):
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.sentence_embedder = DummySentenceEmbedder(max_len=max_len, embed_dim=embed_dim)
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
        self.iter_dataloader = zip(self.articles, self.summaries)
        return self

    def __next__(self):
        article, summary = next(self.iter_dataloader)
        article_sentences = article.split(".")
        embeddings = self.sentence_embedder.embed_sentences(article_sentences)
        return article_sentences, embeddings, summary


class BatchCNNDMLoader(DataLoader):
    def __init__(self, path_to_batches: str = 'data/finished_files/train/', max_len: int = 100, embed_dim: int = 768):
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.path_to_batches = path_to_batches


    @staticmethod
    def precompute_embeddings(path_to_binary: Path, path_to_batches: Path,
                              batch_size: int = 100,
                              bert_model_dir: str = "bert/uncased_L-12_H-768_A-12"):
        """Precompute and store sentence embeddings, along with articles and summaries, in batches of 100.


        :param path_to_binary:  Path to pre-tokenized binaries (train/valid/test.bin,
                                from https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail)
        :param path_to_batches: Path prefix for storing batches of embeddings/articles/summaries
        :param batch_size:      Number of articles per batch.
        :param bert_model_dir:  Directory of bert model (https://github.com/hanxiao/bert-as-service)
        """
        from bert_serving.client import BertClient
        from tensorflow.core.example import example_pb2
        import struct
        import nltk
        nltk.download('punkt')

        # load bert client (and bert server if not already running)
        try:
            bc = BertClient()
        except:
            from bert_serving.server.helper import get_args_parser
            from bert_serving.server import BertServer

            args = get_args_parser().parse_args(['-model_dir', bert_model_dir,
                                                 '-max_seq_len', 40,
                                                 '-num_worker', 4,
                                                 '-device_map', '1,2,3,4'])
            server = BertServer(args)
            server.start()
            bc = BertClient()
        print("Bert client loaded...")

        # load articles and summaries
        articles: List[List[str]] = []
        summaries: List[str] = []
        reader = open(path_to_binary, 'rb')
        i = 0
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
                    i += 1
                    if not i % 1000: print(f"loaded {i} articles...")
            except ValueError:
                print("Failed retrieving an article or abstract.")
        print(f"Articles and summaries read from path: {path_to_binary}...")

        # precompute embeddings, and store batches of embeddings/articles/summaries
        for i in tqdm(range(0, len(articles), batch_size)):
            j = min(len(articles), i + batch_size)
            print(f"embedding articles {i}-{j}...")
            a = articles[i:j]
            s = summaries[i:j]
            articles_tensor = bc.encode(sum(a, []))

            np.savez_compressed(f"{path_to_batches}.article_tensors.{i}.npz", articles_tensor)

            with open(f"{path_to_batches}.sentencized_articles.{i}.pkl", 'wb') as f:
                pickle.dump(a, f)

            with open(f"{path_to_batches}.summaries.{i}.pkl", 'wb') as f:
                pickle.dump(s, f)

            i += batch_size

    def _load_data(self, batch_paths: Tuple[Path, Path, Path]) -> Tuple[List[List[str]], List[np.ndarray], List[str]]:
        """Loads batch of articles/tensors/summaries data.

        :param batch_paths: Paths to batches of articles/tensors/summaries data
        :return:            Tuple of (articles sentences, articles sentence embeddings, articles summaries)
        """
        articles_path, embeddings_path, summaries_path = batch_paths
        with open(articles_path, 'rb') as f:
            articles = pickle.load(f)
        with open(summaries_path, 'rb') as f:
            summaries = pickle.load(f)

        # convert concatenated batch-level embedding array, to a list of embedding arrays for given articles in a batch.
        batch_embeddings = np.load(embeddings_path)['arr_0']
        embeddings = []
        i1 = 0
        for a in articles:
            i2 = i1 + len(a)
            article_embeddings = SentenceEmbedder.pad_embeddings(batch_embeddings[i1:i2], self.max_len)
            embeddings.append(article_embeddings)
            i1 = i2

        return articles, embeddings, summaries

    def __iter__(self):
        articles_batches = sorted(list(Path(self.path_to_batches).glob("sentencized_articles.*.pkl")))
        tensors_batches = sorted(list(Path(self.path_to_batches).glob("article_tensors.*.npz")))
        summaries_batches = sorted(list(Path(self.path_to_batches).glob("summaries.*.pkl")))
        self.batches = zip(articles_batches, tensors_batches, summaries_batches)
        self.batch = self._load_data(next(self.batches))
        self.i = 0
        return self

    def __next__(self) -> Tuple[List[str], np.ndarray, str]:
        try:
            article = self.batch[0][self.i]
            embedding = self.batch[1][self.i]
            summary = self.batch[2][self.i]
            self.i += 1
            return article, embedding, summary
        except IndexError:
            while True: # skip over broken batches (e.g. train 53500)
                try:
                    self.batch = self._load_data(next(self.batches))
                    self.i = 0
                    article = self.batch[0][self.i]
                    embedding = self.batch[1][self.i]
                    summary = self.batch[2][self.i]
                    self.i += 1
                    return article, embedding, summary
                except StopIteration:
                    raise StopIteration

                except Exception as e:
                    warnings.warn(f"[WARNING] Failed to load batch: {self.batches}\n with exception: {str(e)}")



