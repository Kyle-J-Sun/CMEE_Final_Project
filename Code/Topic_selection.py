#!/usr/bin/env python3

""" LDA Topic Number Selection Program """
__author__ = 'Jinkai Sun (jingkai.sun20@imperial.ac.uk), '
__version__ = '0.0.1'

# Import Packages
from imp import reload
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tool_functions import nlp_ldamodel
import warnings
import sys

def main(argvs):
    df = pd.read_csv("../Data/cleanData.csv")
    pp = nlp_ldamodel.preprocessor(dataset = df)
    pp.remove_stop_words()
    pp.word_stemmer(getSet=False)
    words, dic, corpus = pp.get_corpus()
    m = nlp_ldamodel(words, dic, corpus)
    m.cv_repeat(repeatTimes = 10, topicRange = (5,50), iterations=10, countFrom = argvs[1])
    return 0

if __name__ == '__main__':
    status = main(sys.argv)
    sys.exit(status)
