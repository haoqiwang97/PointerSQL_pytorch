#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 23:48:35 2021

@author: haoqiwang
"""
import argparse
from utils import *
from data import *


def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    
    parser.add_argument('--train_path', type=str, default='data/wikisql_train.dat', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/wikisql_dev.dat', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/wikisql_test.dat', help='path to blind test data')
    parser.add_argument('--decoder_len_limit', type=int, default=20, help='output length limit of the decoder')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    
    # Load training and test data
    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path)
    #dev[1][0]='1-10015132-11 player no. nationality position years^in^toronto school/club^team how many schools did player number no. 3 play at ?'
    #dev[1][1]='1-10015132-11 count school/club^team no. = 3'
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev, test, args.decoder_len_limit)
    # train_data_indexed[1].x='1-1000181-1 state/territory text/background^colour format current^slogan current^series notes what is the current^series where the notes new^series^began^in^june^2011 ?'
    # train_data_indexed[1].x_tok
    # train_data_indexed[1].x_indexed=[2, 3, 4, 5, 6, 7, 8, 11, 16, 12, 7, 17, 12, 8, 18, 19]
    
    # train_data_indexed[1].y='1-1000181-1 select current^series notes = new^series^began^in^june^2011'
    # train_data_indexed[1].y_tok
    # train_data_indexed[1].y_indexed=[3, 4, 9, 5, 7, 10, 2]

"""
# how did I choose decoder_len_limit=20
import numpy as np
length_list = np.zeros(len(train), dtype=int)
for i in range(len(train)):
    length_list[i] = len(train[i][1].split())

max(length_list)
>>>15

length_list2 = np.zeros(len(dev), dtype=int)
for i in range(len(dev)):
    length_list2[i] = len(dev[i][1].split())

max(length_list2)
>>>15
# so I put 20
"""
# TODO: table len, query len, do I need to store these values?
# TODO: maintain a small built-in decoder vocabulary (sized 17)
# TODO: entity linking
# TODO: start embedding layer in encoder 