import torchtext
from collections import Counter
import numpy as np
from tokenizers import Tokenizer


class BaseVocab(torchtext.vocab.Vocab):
    def unk(self):
        if "<unk>" not in self.stoi:
            raise KeyError("<unk> not in vocab")
        return self.stoi["<unk>"]

    def pad(self):
        if "<pad>" not in self.stoi:
            raise KeyError("<pad> not in vocab")
        return self.stoi["<pad>"]

    def sos(self):
        if "<s>" not in self.stoi:
            raise KeyError("<s> not in vocab")
        return self.stoi["<s>"]

    def eos(self):
        if "</s>" not in self.stoi:
            raise KeyError("</s> not in vocab")
        return self.stoi["</s>"]

    def index(self, w):
        assert isinstance(w, str)
        if w in self.stoi:
            return self.stoi[w]
        return self.unk()

    def string(self, i):
        assert i >= 0 and i < len(self.itos)
        return self.itos[i]


class Vocab(BaseVocab):
    def __init__(self, vocab_file, *args, **kwargs):
        tokenizer = Tokenizer.from_file(vocab_file)
        super().__init__(
            tokenizer.get_vocab(), min_freq=1, specials=["<pad>", "<unk>"],
        )


class POSVocab(BaseVocab):
    def __init__(self):
        counter = Counter(
            {
                upos: 1
                for upos in [
                    "ADJ",
                    "ADP",
                    "ADV",
                    "AUX",
                    "CCONJ",
                    "DET",
                    "INTJ",
                    "NOUN",
                    "NUM",
                    "PART",
                    "PRON",
                    "PROPN",
                    "PUNCT",
                    "SCONJ",
                    "SYM",
                    "VERB",
                    "X",
                ]
            }
        )
        super().__init__(counter, min_freq=1, specials=(["<pad>"]))


class DEPVocab(BaseVocab):
    def __init__(self):
        counter = Counter(
            {
                upos: 1
                for upos in [
                    "acl",
                    "advcl",
                    "advmod",
                    "amod",
                    "appos",
                    "aux",
                    "case",
                    "cc",
                    "ccomp",
                    "clf",
                    "compound",
                    "conj",
                    "cop",
                    "csubj",
                    "dep",
                    "det",
                    "discourse",
                    "dislocated",
                    "expl",
                    "fixed",
                    "flat",
                    "goeswith",
                    "iobj",
                    "list",
                    "mark",
                    "nmod",
                    "nsubj",
                    "nummod",
                    "obj",
                    "obl",
                    "orphan",
                    "parataxis",
                    "punct",
                    "reparandum",
                    "root",
                    "vocative",
                    "xcomp",
                ]
            }
        )
        super().__init__(counter, min_freq=1, specials=(["<pad>"]))
