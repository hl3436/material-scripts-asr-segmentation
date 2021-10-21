import os
import sys
import random
import re
#import pandas as pd
import numpy as np
import string

from collections import defaultdict

in_dir = sys.argv[1]
work_dir = "/work/raw"

# id: lines
out = {}

for file in os.listdir(in_dir):
    file_prefix = file.split(".")[0]
    #print(file_prefix)
    # read ctm and utt
    if file.endswith(".ctm"):
        utterances = defaultdict(list)

        utt_start = defaultdict(list)

        for line in open(os.path.join(in_dir, file_prefix + ".utt")):
            line = line.strip().split()
            utt_start[line[1]].append(float(line[2]))

        utterance = defaultdict(list)
        for i, line in enumerate(open(os.path.join(in_dir, file))):
            line = line.strip().split()
            speaker = line[1]
            if not utt_start[speaker] or float(line[2]) < utt_start[speaker][0]:
                utterance[speaker].append(line[4])
            else:  # end of sentence or sos
                if utterance[speaker]:
                    utterances[speaker].append(" ".join(utterance[speaker]))
                utterance[speaker] = [line[4]]
                utt_start[speaker] = utt_start[speaker][1:]

        # remaining
        for speaker, utt in utterance.items():
            if utt:
                utterances[speaker].append(" ".join(utt))
        for speaker in utterances:
            out[file_prefix + "_" + speaker + ".txt"] = utterances[speaker]

    elif file.endswith(".rest.bst"):
        suffix = ".".join(file.split(".")[1:])
        documents = []
        lines = []
        for line in open(os.path.join(in_dir, file)):
            line = line.strip()
            if line:
                lines.append(line)
            else:  # end of document
                if lines:
                    documents.append(lines)
                    lines = []
        for i, doc in enumerate(documents):
            out[file_prefix + "_" + str(i) + "." + suffix] = doc

with open(os.path.join(work_dir, "raw.txt"), "w") as f, open(
    os.path.join(work_dir, "raw.tokens"), "w"
) as g, open(os.path.join(work_dir, "raw.id"), "w") as h:
    for file_id, lines in out.items():
        h.write("{}\n".format(file_id))
        for line in lines:
            f.write("{}\n".format(line))
        f.write("\n")
        g.write("{}\n".format(" ".join(lines)))
