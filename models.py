#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 23:27:11 2021

@author: haoqiwang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var
from utils import *
from data import *
# from lf_evaluator import *
import numpy as np
from typing import List
from torch import optim


def add_models_args(parser):
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    
    parser.add_argument('--decoder_len_limit', type=int, default=20, help='output length limit of the decoder')
    parser.add_argument('--emb_dim', type=int, default=200, help='word embedding dimensions')
    parser.add_argument('--hidden_dim', type=int, default=100, help='hidden layer size')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers')

"""
# how did I choose decoder_len_limit=20
import numpy as np
length_list = np.zeros(len(train_exs), dtype=int)
for i in range(len(train)):
    length_list[i] = len(train[i][1].split())

for i in range(len(train_exs)):
    length_list[i] = len(train_exs[i][0].split())
    
max(length_list)
>>>15

length_list2 = np.zeros(len(dev), dtype=int)
for i in range(len(dev)):
    length_list2[i] = len(dev[i][1].split())

max(length_list2)
>>>15
# so I put 20
"""

class Seq2Seq(nn.Module):
    def __init__(self, word_vectors, input_indexer, output_indexer, emb_dim, hidden_dim, bidirect=True, n_layers=3):
        super(Seq2Seq, self).__init__()
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.word_vectors = word_vectors
        
        self.input_emb = EmbeddingLayer(word_vectors)
        self.encoder = Encoder(emb_dim, hidden_dim, bidirect, n_layers)
        
        self.output_emb = EmbeddingLayer(word_vectors)
        
    def forward(self, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor):
        teacher_forcing_value = 0.5
        batch_size = x_tensor.size()[0] 
        
        output_lens = out_lens_tensor.item()
        
        (enc_output_each_word, enc_context_mask, enc_final_states_reshaped) = self.encode_input(x_tensor, inp_lens_tensor)
        
        h_t = enc_final_states_reshaped
        
        fc_outputs = torch.zeros(output_lens, batch_size, len(self.output_indexer))
        pass
    
    def decode(self):
        pass

    def encode_input(self, x_tensor, inp_lens_tensor):
        input_emb = self.input_emb.forward(x_tensor)
        (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(input_emb, inp_lens_tensor)
        enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
        return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)
    
    
class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """
    # def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
    def __init__(self, pretrained_word_vectors):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        # self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_word_vectors.vectors))
        # self.word_embedding = nn.Embedding.from_pretrained(word_vectors)
    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        embedded_words = self.word_embedding(input)
        # final_embeddings = self.dropout(embedded_words)
        return embedded_words#final_embeddings
    
    
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirect, n_layers):
        super(Encoder, self).__init__()
        self.bidirect = bidirect
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reduce_h_W = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.reduce_c_W = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers,
                           batch_first=True, dropout=0, 
                           bidirectional=self.bidirect)
        self.init_weight()
        pass

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, embedded_words, input_lens):
        """
        Runs the forward pass of the LSTM
        :param embedded_words: [batch size x sent len x input dim] tensor
        :param input_lens: [batch size]-length vector containing the length of each input sentence
        :return: output (each word's representation), context_mask (a mask of 0s and 1s
        reflecting where the model's output should be considered), and h_t, a *tuple* containing
        the final states h and c from the encoder for each sentence.
        """
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True, enforce_sorted=False)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        # max_length = input_lens.data[0].item()
        max_length = input_lens.max().item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1] # h & c = (num_layers * num_directions, batch, hidden_size)
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1) # h[0] = [batch, hidden_size], h_ & c_ = [batch, hidden_size*num_directions]
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_) # new_h = [batch, hidden_size]
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        return (output, context_mask, h_t)

    
class DecoderAttention(nn.Module):
    def __init__(self):
        pass

class DecoderCopy(nn.Module):
    def __init__(self):
        pass


def train_model(word_vectors, train_data: List[Example], dev_data: List[Example], input_indexer, output_indexer, args):
    
    # AttributeError: 'tuple' object has no attribute 'words'
    # all_train_input_data
    # all_test_input_data
    
    # all_train_output_data
    # all_test_output_data
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data])) # input_max_len = 19
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False)
    all_test_input_data = make_padded_input_tensor(dev_data, input_indexer, input_max_len, reverse_input=False)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(dev_data, output_indexer, output_max_len)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # Format data
    all_input_lens = torch.from_numpy(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = torch.from_numpy(all_train_input_data)
    all_test_input_data = torch.from_numpy(all_test_input_data)
    
    all_output_lens = torch.from_numpy(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = torch.from_numpy(all_train_output_data)
    all_test_output_data = torch.from_numpy(all_test_output_data)
    
    seq2seq = Seq2Seq(word_vectors, 
                      input_indexer=input_indexer, 
                      output_indexer=output_indexer,
                      emb_dim=args.emb_dim,
                      hidden_dim=args.hidden_dim)
    
    n_epochs = args.epochs
    n_exs = all_train_input_data.size()[0]
    lr = args.lr
    batch_size = args.batch_size
    
    optimizer = optim.Adagrad(seq2seq.parameters(), lr)
    criterion = nn.CrossEntropyLoss(ignore_index = word_vectors.word_indexer.index_of('PAD'))
    
    seq2seq.train()
    for epoch_idx in range(n_epochs):
        ex_indices = [i for i in range(0, n_exs)]
        np.random.shuffle(ex_indices)
        total_loss = 0.0
        for ex_idx in ex_indices:
            x_tensor = all_train_input_data[ex_idx:(ex_idx+batch_size)]
            inp_lens_tensor = all_input_lens[ex_idx:(ex_idx+batch_size)] # input_lens = [batch_size]
            y_tensor = all_train_output_data[ex_idx:(ex_idx+batch_size)]
            out_lens_tensor = all_output_lens[ex_idx:(ex_idx+batch_size)]
            
            optimizer.zero_grad()
            
    return seq2seq
