import sys
import numpy as np
import torchtext
from collections import Counter
from tokenizers import Tokenizer
import os
import torch
import pickle


dir = sys.argv[1]
prefix = sys.argv[2]
tokenizer_file = sys.argv[3]


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
    def __init__(self, json_file, *args, **kwargs):
        tokenizer = Tokenizer.from_file(json_file)
        super().__init__(tokenizer.get_vocab(), min_freq=1, specials=["<pad>", "<unk>"])


def to_bin(in_dir, file_prefix, vocab):
    out_dir = in_dir + "/bin"
    data = []
    for i, line in enumerate(open("{}/{}.tokens".format(in_dir, file_prefix))):
        line = line.strip().split()
        data.append(torch.LongTensor([vocab.index(w) for w in line]))
    np.save(
        "{}/{}.tokens.npy".format(out_dir, file_prefix), np.array(data, dtype=object)
    )
    data = []
    for i, line in enumerate(open("{}/{}.prev_labels".format(in_dir, file_prefix))):
        line = line.strip().split()
        data.append(torch.LongTensor([int(w) for w in line]))
    np.save(
        "{}/{}.prev_labels.npy".format(out_dir, file_prefix),
        np.array(data, dtype=object),
    )


vocab = Vocab(tokenizer_file)

to_bin(dir, prefix, vocab)
