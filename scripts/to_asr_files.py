import pickle
import os
import numpy as np
from tokenizers import Tokenizer
import re
from sacremoses import MosesDetokenizer
import sys

out_dir = sys.argv[1]
lang = sys.argv[2]

token_file = "/work/asr.tokens"
id_file = "/work/asr.id"
prev_labels_file = "/work/asr.prev_labels"
asr_file = "/work/raw/raw.tokens"
asr_id_file = "/work/raw/raw.id"
detok = MosesDetokenizer(lang=lang)
tokenizer = Tokenizer.from_file("/models/{}_tokenizer.json".format(lang))


def split_sentence(tokens, labels, preds):
    split_idx = labels.index("1") + 1
    first_sent = tokens[:split_idx]
    second_sent = tokens[split_idx:]
    first_pred = preds[:split_idx]
    second_pred = preds[split_idx : split_idx + len(second_sent)]
    return first_sent, second_sent, first_pred, second_pred


def decode_s(s, detok):
    s = "".join(s)
    s = re.sub("<unk>", "<unk> ", s)
    s = re.sub("</w>", " ", s)
    s = detok.detokenize(s.split())
    # postprocssing
    s = re.sub("< ", "<", s)
    s = re.sub(" >", ">", s)
    s = re.sub(" ?= ?", "=", s)
    s = re.sub(" ?- ?", "-", s)
    s = re.sub(" ?_ ?", "_", s)
    return s.strip()


with open("/work/pred.data", "rb") as f:
    pred = pickle.load(f)

tokens = [line.strip().split() for line in open(token_file)]
ids = [line.strip() for line in open(id_file)]

prev_labels = [line.strip().split() for line in open(prev_labels_file)]

asr = [line.strip().split() for line in open(asr_file)]

asr_ids = [line.strip() for line in open(asr_id_file)]
asr_dict = {id: token for id, token in zip(asr_ids, asr)}

s1, s2, p1, p2 = split_sentence(tokens[0], prev_labels[0], pred[0])
id = ids[0]
out = {"tokens": [s1, s2], "labels": [[p1], [p2]]}
asr_idx = 0
_asr = asr[asr_idx]
for i, p in enumerate(pred):
    if i == 0:
        continue
    _id = ids[i]
    _s1, _s2, _p1, _p2 = split_sentence(tokens[i], prev_labels[i], pred[i])
    if _id != id:
        out["labels"] = [
            [int(_l) for _l in np.logical_or(*l)] if len(l) > 1 else l[0]
            for l in out["labels"]
        ]

        # flatten tokens and labels
        out["tokens"] = [item for sublist in out["tokens"] for item in sublist]
        out["labels"] = [item for sublist in out["labels"] for item in sublist]

        assert len(out["tokens"]) == len(out["labels"])

        _asr = asr_dict[id]
        if len(out["tokens"]) < len(_asr):
            print(id, len(out["tokens"]), len(_asr))
        assert len(out["tokens"]) >= len(_asr)

        with open(os.path.join(out_dir, id), "w") as f:
            out_tokens = []

            cur_tok = []

            asr_tok_idx = 0
            for j in range(len(out["tokens"])):
                tok = out["tokens"][j]
                lab = out["labels"][j]
                asr_tok = _asr[asr_tok_idx]

                # check if the tok matches the asr tok else continue to add until the token is the asr token
                cur_tok.append(tok)
                cur_tok_detok = tokenizer.decode(
                    [tokenizer.token_to_id(tok) for tok in cur_tok]
                )
                # if id == "MATERIAL_OP2-3S_38272753_1.txt":
                #    print(cur_tok_detok)
                cur_tok_detok = decode_s(cur_tok_detok, detok)
                # if id == "MATERIAL_OP2-3S_67820160_B.txt":
                #    print(cur_tok_detok)
                if cur_tok_detok == asr_tok:
                    # print("match", asr_tok)
                    out_tokens.append(asr_tok)
                    if lab:
                        f.write("{}\n".format(" ".join(out_tokens)))
                        out_tokens = []
                    asr_tok_idx += 1
                    cur_tok = []
                # else:
                #    if "67820160" in id:
                #        print("ERROR", i, j, id, cur_tok_detok, asr_tok)
                # for k in range(len(out["tokens"])):
                #    print(out["tokens"][k], _asr[k])
                # print(aeounaeu)
            if tok:
                cur_tok_detok = tokenizer.decode(
                    [tokenizer.token_to_id(tok) for tok in cur_tok]
                )
                out_tokens.append(cur_tok_detok)

            if out_tokens:
                f.write("{}\n".format(" ".join(out_tokens)))

            asr_idx += 1
            _asr = asr[asr_idx]

        # clear stuff
        id = _id
        out = {"tokens": [_s1, _s2], "labels": [[_p1], [_p2]]}
    else:
        assert out["tokens"][-1] == _s1
        out["tokens"].append(_s2)
        out["labels"][-1].append(_p1)
        out["labels"].append([_p2])

# last file

out["labels"] = [
    [int(_l) for _l in np.logical_or(*l)] if len(l) > 1 else l[0] for l in out["labels"]
]
out["tokens"] = [item for sublist in out["tokens"] for item in sublist]
out["labels"] = [item for sublist in out["labels"] for item in sublist]
_asr = asr_dict[id]
with open(os.path.join(out_dir, id), "w") as f:
    out_tokens = []
    cur_tok = []
    asr_tok_idx = 0
    for j in range(len(out["tokens"])):
        tok = out["tokens"][j]
        lab = out["labels"][j]
        asr_tok = _asr[asr_tok_idx]

        # check if the tok matches the asr tok else continue to add until the token is the asr token
        cur_tok.append(tok)
        cur_tok_detok = tokenizer.decode(
            [tokenizer.token_to_id(tok) for tok in cur_tok]
        )
        cur_tok_detok = decode_s(cur_tok_detok, detok)
        if cur_tok_detok == asr_tok:
            out_tokens.append(cur_tok_detok)
            if lab:
                f.write("{}\n".format(" ".join(out_tokens)))
                out_tokens = []
            asr_tok_idx += 1
            cur_tok = []
    if tok:
        cur_tok_detok = tokenizer.decode(
            [tokenizer.token_to_id(tok) for tok in cur_tok]
        )
        out_tokens.append(cur_tok_detok)

    if out_tokens:
        f.write("{}\n".format(" ".join(out_tokens)))
