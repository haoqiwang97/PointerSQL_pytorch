#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:54:42 2021

@author: haoqiwang
"""

import numpy as np


train_path = 'data/wikisql_train.dat'
dev_path = 'data/wikisql_dev.dat'

n_train_small = 500
n_dev_small = 125


with open(train_path) as f:
    raw_lines = f.readlines()
    n_lines = len(raw_lines)
    # each example takes 4 lines
    # 1st, table and column names, eg, raw_lines[0]='1-10015132-11 player no. nationality position years^in^toronto school/club^team\n'
    # 2nd, query, eg, raw_lines[1]='what position does the player who played for school/club^team butler^cc^(ks) play ?\n'
    # 3rd, y, ground truth, eg, raw_lines[2]='1-10015132-11 select position school/club^team = butler^cc^(ks)\n'
    # 4th, '\n', empty, eg, raw_lines[3]='\n'
    n_examples = n_lines // 4
    np.random.rand(1234)
    train_small_ids = np.random.choice(n_examples, n_train_small, replace=False)
    for example_idx in train_small_ids:
        # print(example_idx)
        f_small = open("data/wikisql_train_small.dat", "a")
        
        f_small.write(raw_lines[example_idx * 4])
        f_small.write(raw_lines[example_idx * 4 + 1])
        f_small.write(raw_lines[example_idx * 4 + 2])
        f_small.write('\n')
        f_small.close()
        

with open(dev_path) as f:
    raw_lines = f.readlines()
    n_lines = len(raw_lines)
    # each example takes 4 lines
    # 1st, table and column names, eg, raw_lines[0]='1-10015132-11 player no. nationality position years^in^toronto school/club^team\n'
    # 2nd, query, eg, raw_lines[1]='what position does the player who played for school/club^team butler^cc^(ks) play ?\n'
    # 3rd, y, ground truth, eg, raw_lines[2]='1-10015132-11 select position school/club^team = butler^cc^(ks)\n'
    # 4th, '\n', empty, eg, raw_lines[3]='\n'
    n_examples = n_lines // 4
    np.random.rand(2345)
    train_small_ids = np.random.choice(n_examples, n_dev_small, replace=False)
    for example_idx in train_small_ids:
        # print(example_idx)
        f_small = open("data/wikisql_dev_small.dat", "a")
        
        f_small.write(raw_lines[example_idx * 4])
        f_small.write(raw_lines[example_idx * 4 + 1])
        f_small.write(raw_lines[example_idx * 4 + 2])
        f_small.write('\n')
        f_small.close()