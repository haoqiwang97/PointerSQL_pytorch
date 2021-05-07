#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 23:48:35 2021

@author: haoqiwang
"""
import argparse
from utils import *
from data import *
from models import *


def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    
    parser.add_argument('--train_path', type=str, default='data/wikisql_train_small.dat', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/wikisql_dev_small.dat', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/wikisql_test.dat', help='path to blind test data')
    # parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.100d.txt', help='path to word embeddings to use')
    parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.50d-relativized.txt', help='path to word embeddings to use')
    parser.add_argument('--n_gram_path', type=str, default='data/jmt_char_n_gram.txt', help='path to ngrams to use')
    parser.add_argument('--use_pretrained', type=bool, default=False, help='use pretrained word vector or not')
    
    add_models_args(parser) # defined in models.py
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    
    # Load training and test data
    train_exs, dev_exs, test_exs = load_datasets(args.train_path, args.dev_path, args.test_path)
    #dev[1][0]='1-10015132-11 player no. nationality position years^in^toronto school/club^team how many schools did player number no. 3 play at ?'
    #dev[1][1]='1-10015132-11 count school/club^team no. = 3'
    if args.use_pretrained == True:
        word_vectors = load_word_vecs(args.word_vecs_path)
        # train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(word_vectors, train_exs, dev_exs, test_exs, args.decoder_len_limit)
    else:
        word_vectors = None
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(word_vectors, train_exs, dev_exs, test_exs, args.decoder_len_limit, use_pretrained=args.use_pretrained)
    # train_data_indexed[1].x='1-1000181-1 state/territory text/background^colour format current^slogan current^series notes what is the current^series where the notes new^series^began^in^june^2011 ?'
    # train_data_indexed[1].x_tok
    # train_data_indexed[1].x_indexed=[2, 3, 4, 5, 6, 7, 8, 11, 16, 12, 7, 17, 12, 8, 18, 19]
    
    # train_data_indexed[1].y='1-1000181-1 select current^series notes = new^series^began^in^june^2011'
    # train_data_indexed[1].y_tok
    # train_data_indexed[1].y_indexed=[3, 4, 9, 5, 7, 10, 2]
    
    # word_vecs.word_indexer.get_object(1)='UNK'
    # word_vecs.word_indexer.get_object(100)='against'
    
    # n_gram = load_n_gram(args.n_gram_path)

    # decoder = train_model(word_vectors, train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
    decoder = train_model(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
    
    decoder.decode(train_data_indexed[0:10])
# TODO: maintain a small built-in decoder vocabulary (sized 17)
