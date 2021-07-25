#!/usr/bin/env python3

""" Cleaning NSF, UKRI, ERC and NIH datasets """

## Variables ##

__author__ = 'Jinkai Sun (jingkai.sun20@imperial.ac.uk), '
__version__ = '0.0.1'

import pandas as pd
import numpy as np
import os
import re
from tool_functions_copy import remove_punctuation, MySQLPipline, modelProcessor

import warnings
import sys
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

def main(argvs):
    funding = MySQLPipline(database='funding')
    NIHdata = funding.NIHDataset()
    NSFdata = funding.NSFDataset()
    ERCdata = funding.ERCDataset()
    UKRIdata = funding.UKRIDataset()
    funding.close_Conn()
    
    df1 = pd.concat([NIHdata[["title", "abstract"]],
                    ERCdata[["title", "abstract"]]])
    df2 = pd.concat([NSFdata[["title", "abstract"]],
                    UKRIdata[["title", "abstract"]]])
    df = pd.concat([df1, df2])
    df.reset_index(drop=True, inplace=True)
    
    # Remove projects of which abstract is unavailable
    df = remove_abstract(df, regex=r'Abstracts are not currently available in GtR')
    df = remove_abstract(df, regex=r'No abstract available')

    # Removing Punctuations
    pd.options.mode.chained_assignment = None
    df["abstract"] = df["abstract"].apply(remove_punctuation)
    df["abstract"] = df["abstract"].apply(lambda s: s.replace("ltbrgtltbrgt", " "))
    df["abstract"] = df["abstract"].apply(
        lambda s: s.replace("\ufeff", "").strip())
    df["abstract"] = df["abstract"].apply(lambda x: x.replace(
        "\n", " ").replace("ampquot", "").replace("ampamp", ""))

    # Remove digits and multiple spaces
    while True:
        df['abstract'] = df.abstract.apply(
            lambda s: re.sub(r'\s+[0-9]+\s+', " ", s))
        df['abstract'] = df.abstract.apply(
            lambda s: re.sub(r'\s{2,}', " ", s.strip()))
        no_digits = noDigitsSpace_check(df, 1)
        print(no_digits)
        if no_digits:
            break

    # Removing all project with empty abstract
    df = df[~df["abstract"].isna()]
    df = df[df["abstract"] != ""]
    df = df[df.abstract != "NA"]

    df.reset_index(drop=True, inplace=True)

    # Remove all projects whose the abstract only contains numbers
    df = df[~df.index.isin(multiple_regexFind(
        df, pattern='^[0-9]*$', return_df=False))]

    # Remove all duplicates
    df = df[~df.duplicated()]

    # Reset index
    df.reset_index(drop=True, inplace=True)

    col = df["abstract"].apply(lambda s: True if len(s) <= 70 else False).tolist()
    df = df[~df.index.isin([i for i, x in enumerate(col) if x])]

    # Reset index
    df.reset_index(drop=True, inplace=True)
    
    df.to_csv("../Data/cleaned_data2.csv", index = False, encoding = "utf-8-sig")

    return 0

if __name__ == '__main__':
    status = main(sys.argv)
    sys.exit(status)
