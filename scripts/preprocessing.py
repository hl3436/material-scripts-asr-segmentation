from sacremoses import MosesTokenizer, MosesPunctNormalizer
import sys

raw_file = sys.argv[1]
out_file = sys.argv[2]
lang = sys.argv[3]

mt = MosesTokenizer(lang=lang)
mpn = MosesPunctNormalizer()

with open(out_file, "w") as f:
    for line in open(raw_file):
        line = line.strip().lower()
        line = mpn.normalize(line)
        line = mt.tokenize(line, return_str=True, aggressive_dash_splits=True)
        f.write("{}\n".format(line))
