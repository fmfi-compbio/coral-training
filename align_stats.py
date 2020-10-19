from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import numpy as np

def align_dist(outputs, bby):
    alph = "NACGT"

    ed_total = 0
    size_total = 0
    for out, by in zip(outputs[:3], bby[:3]):
        targets = [x for x in by if x > 0]
        preds = []
        prev = 0
        for p in list(np.argmax(out, axis=-1)):
            if p == prev:
                continue
            if p != 0:
                preds.append(p)
            prev = p
        alignment = pairwise2.align.globalms(
            "A" + "".join(map(lambda x: alph[x], targets)),
            "A" + "".join(map(lambda x: alph[x], preds)),
            0,
            -1,
            -1,
            -1,
        )
        alignment = alignment[0]

        ed_total -= alignment[2]
        size_total += max(len(targets), len(alignment[0]) - 1)
    return ed_total, size_total