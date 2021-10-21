import sys
from tokenizers import Tokenizer
import os

dir = sys.argv[1]
lang = sys.argv[2]


def tokenize(file):
    _file = file.split(".")
    with open(os.path.join(dir, "bpe.tokens"), "w") as f:
        for line in open(file):
            line = line.strip().split()
            enc = tokenizer.encode(line, is_pretokenized=True)
            tok = " ".join(enc.tokens)
            f.write(tok)
            f.write("\n")


tokenizer = Tokenizer.from_file("/models/{}_tokenizer.json".format(lang))
tokenize(os.path.join(dir, "tok.tokens"))

# apply bpe to prev_labels
bpe_tokens = []
emptyline = set()
for i, line in enumerate(open(os.path.join(dir, "bpe.tokens"))):
    line = line.strip()
    bpe_tokens.append(line)

data = [
    line.strip()
    for i, line in enumerate(open(os.path.join(dir, "tok.prev_labels")))
    if i not in emptyline
]
with open(os.path.join(dir, "bpe.prev_labels"), "w") as f:
    for i in range(len(bpe_tokens)):
        tok = bpe_tokens[i].split()
        lab = data[i].split()

        bpe_labels = []
        lab_idx = 0
        for t in tok:
            if "</w>" not in t:
                bpe_labels.append("0")
            else:
                bpe_labels.append(lab[lab_idx])
                lab_idx += 1

        f.write("{}\n".format(" ".join(bpe_labels)))

with open(os.path.join(dir, "bpe.id"), "w") as f:
    for line in open(os.path.join(dir, "raw.id")):
        f.write(line)
