import numpy as np
import sys
import random
import os
import math
import functools

def med_mad(x, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    """
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad

def my_listdir(d, k=200):
    rng = random.Random(4247)
    files = sorted(os.listdir(d))
    if len(files) < k:
        return [os.path.join(d, f) for f in files]
    return [os.path.join(d, f) for f in rng.sample(files, k)]


def partition(iter, pred):
    _true = []
    _false = []
    for x in iter:
        if pred(x):
            _true.append(x)
        else:
            _false.append(x)
    return (_true, _false)


def get_dataset_files(root_dir):
    #return my_listdir(root_dir, 5000)
    rng = random.Random(47)
    all_file_list = my_listdir(root_dir, 500000)
    pcr_files, native_files = partition(all_file_list, lambda fname: "GXB" in fname or "MINICOL" in fname)

    file_list = rng.sample(pcr_files, 5000) + native_files
    return file_list

def load_file(fname):
    alph = np.zeros(256, dtype=np.int8)
    for i,c in enumerate("ACGTN"):
        alph[ord(c)] = i

    data = np.load(fname)

    x = np.array(data[data.files[0]], dtype=np.float32)
    med, mad = med_mad(x)

    x -= med
    x /= mad
    y_raw = data[data.files[1]]
    y = np.array(alph[y_raw.view(dtype=np.uint32)])

    return x, y

def load_files(files, min_len=5000, drop_sparse=False, shift=0):
    print("loading", len(files), "files")
    X = []
    Y = []
    for i, fn in enumerate(files):
        x, y = load_file(fn)

        if len(x) > min_len:
            X.append(x)
            Y.append(y)
            
        if i % 100 == 99:
            print("done", i, "of", len(files), "good", len(X))
            sys.stdout.flush()
    return X, Y

def load_dir(data_dir):
    files = get_dataset_files(data_dir)
    return load_files(files)

def prep_batch(
    X,
    Y,
    rng,
    batch_size=3,
    leng=250,
    min_ratio=None,
    target_cut=10,
    clip=True,
):
    bx = []
    by = []
    while len(bx) < batch_size:
        sam = rng.randrange(len(X))
        #print(sam)
        if len(X[sam]) < leng + 100:
            continue
        pos = rng.randrange(len(X[sam]) - leng - 50)

        x = np.array(X[sam][pos : pos + leng], dtype=np.float32)
        if clip:
            x = np.clip(x, -2.5, 2.5)
        x = np.vstack([x]).T
        y_base = Y[sam][pos:pos+leng][target_cut:-target_cut]

        y = [point for point in y_base if point != 4]

        if len(y) < 10 or len(y) > x.shape[0] // 3:
            continue
            
        if min_ratio and len(y) * min_ratio <= len(x) - 2 * target_cut:
            continue

        bx.append(x)
        by.append(np.pad(y, (0, leng-len(y)), constant_values=-1))

    return (
        bx,
        by
    )
