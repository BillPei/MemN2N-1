from collections import Counter
from string import digits

import os


def preprocess_data(fname):
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise Exception("[!] Data file {} not found".format(fname))

    count = []
    vocab = {}

    words = []
    curr_idx = 0
    high_idx = 0
    for line in lines:
        pline = line.replace(".", " ").replace("?", " ").lower().split()

        idx = int(pline[0])
        if idx == 1:
            curr_idx = 0
        if "?" not in line:
            curr_idx += 1
        if curr_idx > high_idx:
            high_idx = curr_idx

        words.extend(pline)

    count.append(("<go>", 0))
    count.extend(Counter(words).most_common())
    words, _ = list(zip(*count))
    vocab = dict(zip(words, range(len(words))))

    return count, vocab, high_idx


def read_data(fname, vocab):
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise Exception("[!] Data file {} not found".format(fname))

    for line in lines:
        question = False
        idx = int(line.split()[0])
        line = line.translate(None, digits)

        if idx == 1:
            x = []

        if "?" in line:
            question = True
            line = line.replace(".", " ").replace("?", " ").lower()
            qa = line.split("\t")
            q = []
            a = []
            for word in qa[0].split():
                q.append(vocab[word])
            for word in qa[1].split():
                a.append(vocab[word])
        else:
            line = line.replace(".", " ").replace("?", " ").lower()
            x.append(vocab["<go>"])
            for word in line.split():
                x.append(vocab[word])

        if question:
            yield (x, q, a)
