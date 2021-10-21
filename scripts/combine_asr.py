import os
import sys

asr_dir = sys.argv[1]
ctm_dir = sys.argv[2]
out_dir = sys.argv[3]


def get_order(starts):
    # starts is a dict with corresponding start time
    a_idx, b_idx = 0, 0
    order = []
    while a_idx + b_idx < len(starts["A"]) + len(starts["B"]):
        sel_a, sel_b = False, False
        if a_idx == len(starts["A"]):
            sel_b = True
        elif b_idx == len(starts["B"]):
            sel_a = True
        else:
            if starts["A"][a_idx] <= starts["B"][b_idx]:
                sel_a = True
            else:
                sel_b = True

        # process
        if sel_a:
            order.append(("A", a_idx))
            a_idx += 1
        else:
            order.append(("B", b_idx))
            b_idx += 1
    assert len(order) == len(starts["A"]) + len(starts["B"])
    return order


for file in os.listdir(asr_dir):
    if file.endswith("_1.txt"):

        ctm = []
        starts, ends = [], []
        for line in open(os.path.join(ctm_dir, file[:-6] + ".ctm")):
            line = line.strip().split()
            ctm.append((line[4], float(line[2]), float(line[3])))

        with open(os.path.join(out_dir, file[:-6] + ".utt"), "w") as f, open(
            os.path.join(out_dir, file[:-6] + ".txt"), "w"
        ) as g:
            for line in open(os.path.join(asr_dir, file)):
                line = line.strip().split()
                if not line:
                    continue

                start = ctm[0][1]
                end = ctm[0][1] + ctm[0][2]
                assert line[0] == ctm[0][0]

                ctm = ctm[1:]

                for tok in line[1:]:
                    # print(file, ctm[0], tok)
                    assert ctm[0][0] == tok
                    end = ctm[0][1] + ctm[0][2]
                    ctm = ctm[1:]

                f.write(
                    "{} {} {} {} {}\n".format(
                        file[:-6], "1", start, end, " ".join(line)
                    )
                )
                g.write("{}\n".format(" ".join(line)))

    elif file.endswith("_A.txt"):
        a_lines = [line.strip().split() for line in open(os.path.join(asr_dir, file))]
        b_lines = [
            line.strip().split()
            for line in open(os.path.join(asr_dir, file[:-5] + "B.txt"))
        ]
        # load ctm
        ctm = {"A": [], "B": []}
        for line in open(os.path.join(ctm_dir, file[:-6] + ".ctm")):
            line = line.strip().split()
            ctm[line[1]].append((line[4], float(line[2]), float(line[3])))
        # get starts for a and b
        starts = {"A": [], "B": []}
        starts_time = {"A": [], "B": []}
        end = {"A": [], "B": []}
        lines = {"A": a_lines, "B": b_lines}
        # print(lines)
        # print(ctm)
        for speaker in starts:
            # print(speaker)
            cur_lines = lines[speaker]
            _ctm = ctm[speaker]
            for line in cur_lines:
                if not line:
                    continue
                # print("line", line)
                # print("ctm", _ctm[0])
                # assert not line or _ctm[0][0] == line[0]
                if line and _ctm[0][0] != line[0]:
                    print(file, speaker)
                    print(_ctm)
                    print(lines)
                    # print(neauhase)

                    print(speaker, line[0], _ctm[0])

                starts[speaker].append(_ctm[0][0])

                starts_time[speaker].append(_ctm[0][1])
                _dur = _ctm[0][1] + _ctm[0][2]

                _ctm = _ctm[1:]
                for tok in line[1:]:
                    # print(tok, _ctm[0])
                    # assert not tok or _ctm[0][0] == tok
                    if tok and _ctm[0][0] != tok:
                        print(file)
                        print(speaker, tok, _ctm[0])
                    if _ctm:
                        _dur = _ctm[0][1] + _ctm[0][2]
                    # print(tok,_ctm[0])
                    _ctm = _ctm[1:]
                end[speaker].append(_dur)
        # print(starts)
        order = get_order(starts_time)
        # print(order)
        # print(order)
        # to utt file
        asr_lines = {"A": a_lines, "B": b_lines}
        with open(os.path.join(out_dir, file[:-6] + ".utt"), "w") as f, open(
            os.path.join(out_dir, file[:-6] + ".txt"), "w"
        ) as g:
            for _speaker, _idx in order:
                f.write(
                    "{} {} {:.2f} {:.2f} {}\n".format(
                        file[:-6],
                        _speaker,
                        starts_time[_speaker][_idx],
                        end[_speaker][_idx],
                        " ".join(asr_lines[_speaker][_idx]),
                    )
                )
                g.write("{}\n".format(" ".join(asr_lines[_speaker][_idx])))

    elif file.endswith("_0.rest.bst"):
        file_prefix = file[:-10]
        lines = []
        idx = 0
        while os.path.isfile(
            os.path.join(asr_dir, file_prefix + str(idx) + ".rest.bst")
        ):
            lines.append(
                open(
                    os.path.join(asr_dir, file_prefix + str(idx) + ".rest.bst")
                ).readlines()
            )
            idx += 1

        with open(os.path.join(out_dir, file_prefix[:-1] + ".rest.bst"), "w") as f:
            for line in lines:
                f.writelines(line)
                f.write("\n")
