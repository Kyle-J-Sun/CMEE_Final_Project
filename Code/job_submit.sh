#!/bin/bash

# Author: Jingkai Sun
# Script: job_submit.sh
# Desc: Submission Commends for High Performance Computing at Imperial College London
# Arguments: none
# Date: Jul 2021

#PBS -l walltime=60:00:00
#PBS -l select=1:ncpus=1:mem=1gb
#PBS -J 1-15

module load anaconda3/personal
source activate python_prj

# Install packages
echo "Installing Python packages..."
# pip3 install pandas
# pip3 install numpy
# pip3 install gensim
# pip3 install matplotlib
# pip3 install tqdm
# pip3 install zhon
# pip3 install nltk

echo "Python is about to run"
python3 Topic_selection_HPC.py

echo "Python has finished running"