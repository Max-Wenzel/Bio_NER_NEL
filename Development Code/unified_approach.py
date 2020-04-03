from seqlearn.datasets import load_conll
from seqlearn.perceptron import StructuredPerceptron
from seqlearn.evaluation import bio_f_score
from seqlearn.hmm import MultinomialHMM
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import sklearn as sk
import nltk as nl
import os
import re
from file_loader import create_file


def features(seq, i):
    p = seq[i].split()[0]
    pos = seq[i].split()[1]
    yield "word=" + p  # Current word and POS
    yield "pos=" + pos
    if i == 0:  # add info for the previous word
        yield "preword=" + "START"
        yield "prepos=" + "START"
    else:
        pp = seq[i-1].split()[0]
        ppos = seq[i-1].split()[1]
        yield "prepos=" + ppos
        if pp.isupper() and len(pp) == 3:  # check if previous word is acronym
            yield "preUpper"
        if re.search(r"\d", pp.lower()):  # check if prev word has number
            yield "preNumber"
        yield "preword=" + pp.lower()
    if (i + 1) == len(seq):
        yield "folword=" + "END"
        yield "folpos=" + "END"
    else:  # check the same for the next word
        nnp = seq[i+1].split()[0]
        nnpos = seq[i+1].split()[1]
        yield "folpos=" + nnpos
        if nnp.isupper():
            yield "folUpper"
        if re.search(r"\d", nnp.lower()):
            yield "folNumber"
        yield "folword=" + nnp.lower()
    if p.isupper() and len(p) == 3:
        yield "Uppercase"
    if re.search(r"\d", p.lower()):
        yield "Number"
    if len(p) > 8:  # check if current word is unusually long
        yield "Long"


if __name__ == '__main__':
    train_path = "../Data/bio-ner/train"
    dev_path = "../Data/bio-ner/dev"

    # create_file(train_path, "train")
    # create_file(dev_path, "dev")

    X_train, y_train, l_train = load_conll("train", features)
    X_test, y_test, l_test = load_conll("dev", features)

    per = StructuredPerceptron(lr_exponent=0.15, max_iter=300, verbose=1)
    per.fit(X_train, y_train, l_train)

    y_p = per.predict(X_test, l_test)
    # for x in zip(y_p, y_test):
    #     print(x)

    print(bio_f_score(y_test, y_p))









