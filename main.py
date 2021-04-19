#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 23:48:35 2021

@author: haoqiwang
"""
import argparse
from utils import *

def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    
    parser.add_argument('--train_path', type=str, default='data/wikisql_train.dat', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/wikisql_dev.dat', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/wikisql_test.dat', help='path to blind test data')
    
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    
    # Load training and test data
    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path)