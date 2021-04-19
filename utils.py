#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:01:58 2021

@author: haoqiwang
"""

# import os
from typing import List, Tuple

# train_file = 'data/wikisql_dev.dat'


def load_datasets(train_path: str, dev_path: str, test_path: str) -> (List[Tuple[str, str, str]], List[Tuple[str, str]], List[Tuple[str, str]]):
    train_raw = load_dataset(train_path)
    dev_raw = load_dataset(dev_path)
    test_raw = load_dataset(test_path)
    return train_raw, dev_raw, test_raw


def load_dataset(filename: str) -> List[Tuple[str, str, str]]:
    dataset = []
    with open(filename) as f:
        raw_lines = f.readlines()
        n_lines = len(raw_lines)
        # each example takes 4 lines
        # 1st, table and column names, eg, raw_lines[0]='1-10015132-11 player no. nationality position years^in^toronto school/club^team\n'
        # 2nd, query, eg, raw_lines[1]='what position does the player who played for school/club^team butler^cc^(ks) play ?\n'
        # 3rd, y, ground truth, eg, raw_lines[2]='1-10015132-11 select position school/club^team = butler^cc^(ks)\n'
        # 4th, '\n', empty, eg, raw_lines[3]='\n'
        n_examples = n_lines // 4
        for example_idx in range(n_examples):
            table_column_name = raw_lines[example_idx * 4].strip('\n')
            query = raw_lines[example_idx * 4 + 1].strip('\n')
            y = raw_lines[example_idx * 4 + 2].strip('\n')

            dataset.append((table_column_name, query, y))
    print("Loaded %i examples from file %s" % (n_examples, filename))
    return dataset


# TODO: index data