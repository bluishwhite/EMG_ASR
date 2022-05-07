import json
from .tokenizer import Tokenizer, _Tokenizer


class Vocabulary_CTC(object):
    PAD = 0
    EOS = 1
    BOS = 2
    UNK = 1
    BLANK_ID = 2
    def __init__(self, type, dict_path, max_n_words=-1):

        self.dict_path = dict_path
        self._max_n_words = max_n_words

        self._load_vocab(self.dict_path)
        self._id2token = dict([(ii[0], ww) for ww, ii in self._token2id_feq.items()])
        self.tokenizer = Tokenizer(type=type)  # type: _Tokenizer

    @property
    def max_n_words(self):

        if self._max_n_words == -1:
            return len(self._token2id_feq)
        else:
            return self._max_n_words

    def _init_dict(self):

        return {
            "<PAD>": (self.PAD, 0),
            "<UNK>": (self.UNK, 0),
            "<BLANK_ID>": (self.BLANK_ID, 0)
        }

    def _load_vocab(self, path):
        """
        Load vocabulary from file

        If file is formatted as json, for each item the key is the token, while the value is a tuple such as
        (word_id, word_feq), or a integer which is the index of the token. The index should start from 0.

        If file is formatted as a text file, each line is a token
        """
        self._token2id_feq = self._init_dict()
        N = len(self._token2id_feq)

        if path.endswith(".json"):

            with open(path) as f:
                _dict = json.load(f)
                # Word to word index and word frequence.
                for ww, vv in _dict.items():
                    if isinstance(vv, int):
                        self._token2id_feq[ww] = (vv + N, 0)
                    else:
                        self._token2id_feq[ww] = (vv[0] + N, vv[1])
        else:
            with open(path) as f:
                for i, line in enumerate(f):
                    ww = line.strip().split()[0]
                    self._token2id_feq[ww] = (i + N, 0)

    def token2id(self, word):

        if word in self._token2id_feq and self._token2id_feq[word][0] < self.max_n_words:

            return self._token2id_feq[word][0]
        else:
            return self.UNK

    def id2token(self, word_id):

        return self._id2token[word_id]

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.BOS

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.PAD

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.EOS

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.UNK
    
    def blank_id(self):
        return self.BLANK_ID


PAD = Vocabulary_CTC.PAD
EOS = Vocabulary_CTC.EOS
BOS = Vocabulary_CTC.BOS
UNK = Vocabulary_CTC.UNK
