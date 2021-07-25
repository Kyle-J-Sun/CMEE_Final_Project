#!/usr/bin/env python3

""" LDA Topic Number Selection Program """
__author__ = 'Jinkai Sun (jingkai.sun20@imperial.ac.uk), '
__version__ = '0.0.1'

# Import Packages
from imp import reload
import pandas as pd
import numpy as np
from numpy import array
import os
from tool_functions import cv_repeat, modelProcessor

import warnings
import sys
import time
# To ignore all warnings that arise here to enhance clarity
warnings.filterwarnings('ignore')

def main(argvs):
    df = pd.read_csv("../Data/cleaned_data.csv", encoding='utf-8-sig')
    df = df[0:10]
    mp = modelProcessor(dat=df)
    # Parallel Computing
    ite = int(os.environ['PBS_ARRAY_INDEX'])
    # ite = 3
    if ite == 1:
        count = 0
        while count <= 2:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 2:
        count = 3
        while count <= 5:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 3:
        count = 6
        while count <= 8:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 4:
        count = 9
        while count <= 11:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 5:
        count = 12
        while count <= 14:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 6:
        count = 15
        while count <= 17:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 7:
        count = 18
        while count <= 20:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 8:
        count = 21
        while count <= 23:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 9:
        count = 24
        while count <= 26:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 10:
        count = 27
        while count <= 29:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 11:
        count = 30
        while count <= 32:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 12:
        count = 33
        while count <= 35:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 13:
        count = 36
        while count <= 38:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 14:
        count = 39
        while count <= 41:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    if ite == 15:
        count = 42
        while count <= 44:
            print("starting %sth topic selection..." % str(count))
            grid = mp.topicNumSelection(num_iterations=100, numTopics_range=range(5, 40))
            grid.to_csv("../Results/topicCVs_count{}.csv".format(count), index=False)
            count += 1
    return 0

if __name__ == '__main__':
    status = main(sys.argv)
    sys.exit(status)
