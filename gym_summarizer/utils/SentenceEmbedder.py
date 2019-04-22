import numpy as np
from typing import List
from bert_serving.client import BertClient


class SentenceEmbedder:
    """Helper class for computing pretrained sentence embeddings.

    """
    def __init__(self):
        self.embed_dim: int = None
        self.max_len: int = None

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Computes pretrained sentence embeddings

        :param sentences:   List of sentences to be embedded.
        :return:            Array of corresponding sentence embeddings, with shape (max_len, embed_dim).
        """
        raise NotImplementedError

    @staticmethod
    def pad_embeddings(embeddings: List[np.ndarray], max_len: int = None) -> np.ndarray:
        """Truncate or 0-pad list of sentence embeddings for a given document to specified max length.

        :param embeddings:  List of sentence embeddings for a given document (e.g. news article).
        :param max_len:     Number of sentences to truncate/pad document.
        :return:            Truncated/padded document embeddings.
        """
        if max_len is None:
            return np.array(embeddings)

        padded = np.zeros((max_len, len(embeddings[0])), dtype=np.float32)
        if len(embeddings) > max_len:
            padded[:] = embeddings[:max_len]
        else:
            padded[:len(embeddings)] = embeddings
        return padded


class DummySentenceEmbedder(SentenceEmbedder):
    """ Dummy sentence embedder for testing purposes

    """
    def __init__(self, embed_dim: int = 10, max_len: int = 10):
        self.embed_dim = embed_dim
        self.max_len = max_len

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        embeddings = np.random.randn(len(sentences), self.embed_dim)
        return self.pad_embeddings(embeddings, self.max_len)


class USESentenceEmbedder(SentenceEmbedder):
    def __init__(self, embed_dim: int = 512, max_len: int = 10):
        import tensorflow_hub as hub
        import tensorflow as tf
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        self.se_session = tf.Session()
        self.se_session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        self.embed_dim = embed_dim
        self.max_len = max_len

    def embed_sentences(self, sentences: List[str]) -> List[np.ndarray]:
        embeddings = self.se_session.run(self.embed(sentences))
        return self.pad_embeddings(embeddings, self.max_len)


class BertSentenceEmbedder(SentenceEmbedder):
    def __init__(self, embed_dim: int = 768, max_len: int = 100,
                 model_dir: str = "data/bert/uncased_L-12_H-768_A-12/", max_seq_len: str = '40',
                 device_map: str = '1 2 3 4', num_worker: str = '4'):
        self._init_bert_client(model_dir, max_seq_len, device_map, num_worker)
        self.embed_dim = embed_dim
        self.max_len = max_len

    @staticmethod
    def _init_bert_client(model_dir, max_seq_len, device_map, num_worker) -> BertClient:
        """Initialize bert client for sentence embeddings and avoid restarting bert-server if already running.

        For more information, see: https://github.com/hanxiao/bert-as-service
        Bert-server can take a long time to start, take over stdout during training, and create many temp log files.
        It's highly recommended to run bert-server beforehand from command-line in a dedicated folder:
        e.g:
        ~/gym-summarizer/data/bert $
            bert-serving-start -model_dir uncased_L-12_H-768_A-12/ -max_seq_len 40 -device_map 1 2 3 4 -num_worker 4

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
                                                 '-device_map', device_map,
                                                 '-num_worker', num_worker])
            server = BertServer(args)
            server.start()
            bc = BertClient()

        return bc

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        embeddings = self.bc.encode(sentences)
        return self.pad_embeddings(embeddings, self.max_len)


