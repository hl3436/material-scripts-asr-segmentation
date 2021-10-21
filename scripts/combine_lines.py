import sys
import os

out = dict()

dir = sys.argv[1]

_out = {"lines": [], "labels": []}
file_ids = open(os.path.join(dir, "raw.id")).readlines()
id_idx = 0

for line in open(os.path.join(dir, "tok.txt")):
    line = line.strip().split()
    if line:
        lab = ["0"] * (len(line) - 1) + ["1"]
        assert len(line) == len(lab)
        _out["lines"].append(" ".join(line))
        _out["labels"].append(" ".join(lab))
    else:
        if not _out["lines"]:
            continue
        file_id = file_ids[id_idx].strip()
        out[file_id] = (" ".join(_out["lines"]), " ".join(_out["labels"]))
        _out = {"lines": [], "labels": []}
        id_idx += 1

with open(os.path.join(dir, "tok.tokens"), "w") as f, open(
    os.path.join(dir, "tok.prev_labels"), "w"
) as g:
    for id in out:
        token, label = out[id]
        f.write("{}\n".format(token))
        g.write("{}\n".format(label))
