import os
import sys

in_dir = sys.argv[1]
out_dir = sys.argv[2]


fields = ["tokens", "prev_labels", "id"]

data = {field: [] for field in fields}
for field in fields:
    for line in open(os.path.join(in_dir, "bpe." + field)):
        line = line.strip().split()
        data[field].append(line)

out = {field: [] for field in fields}
for i, id in enumerate(data["id"]):
    lab = data["prev_labels"][i]
    tok = data["tokens"][i]
    idx = lab.index("1")
    first_lab = lab[: idx + 1]
    lab = lab[idx + 1 :]
    first_tok = tok[: idx + 1]
    tok = tok[idx + 1 :]
    second_tok, second_lab = [], []
    for j, l in enumerate(lab):
        second_tok.append(tok[j])
        second_lab.append(l)
        # second_pos.append(pos[j])
        if l == "1":
            # to out and reser first and second stuff
            out["id"].append(id)
            out["tokens"].append(first_tok + second_tok)
            out["prev_labels"].append(first_lab + second_lab)
            first_tok = second_tok
            first_lab = second_lab
            second_tok, second_lab = [], []

for field in fields:
    with open(os.path.join(out_dir, "asr." + field), "w") as f:
        for line in out[field]:
            f.write("{}\n".format(" ".join(line)))
