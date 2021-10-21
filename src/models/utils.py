import json
import torch


def load_pretrained_embedding(embed, vocab, pretrained_file):
    with open(pretrained_file) as f:
        pretrained = json.load(f)
    for i, token in enumerate(vocab.itos):
        if token in pretrained:
            print(embed[i])
            embed[i] = torch.Tensor(pretrained[token])
            print(embed[i])
    return embed
