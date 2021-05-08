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
from math import exp
from torch import autograd
import time


def add_models_args(parser):
    # parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    parser.add_argument('--decoder_len_limit', type=int, default=22, help='output length limit of the decoder')
    parser.add_argument('--emb_dim', type=int, default=100, help='word embedding dimensions')
    parser.add_argument('--embedding_dropout', type=float, default=0.2, help='embedding layer dropout rate')
    parser.add_argument('--hidden_dim', type=int, default=100, help='hidden layer size')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers')


"""
# how did I choose decoder_len_limit=15
import numpy as np
length_list = np.zeros(len(train_exs), dtype=int)
for i in range(len(train)):
    length_list[i] = len(train[i][1].split())

for i in range(len(train_exs)):
    length_list[i] = len(train_exs[i][1].split())
    if len(train_exs[i][1].split()) == 21:
        print(train_exs[i][1].split())
        break
    
max(length_list)
>>>22

length_list2 = np.zeros(len(dev), dtype=int)
for i in range(len(dev)):
    length_list2[i] = len(dev[i][1].split())

max(length_list2)
>>>15
# so I put 15
"""


class Seq2Seq(nn.Module):
    def __init__(self, use_pretrained, word_vectors, input_indexer, output_indexer, emb_dim, hidden_dim,
                 embedding_dropout, bidirect=True, n_layers=3,
                 beam_size=3, decoder_len_limit=20):
        super(Seq2Seq, self).__init__()
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.grammer_indexer = createGrammerIndexer()
        for tok in self.grammer_indexer.objs_to_ints:
            self.output_indexer.add_and_get_index(tok)
        self.word_vectors = word_vectors

        self.input_emb = EmbeddingLayer(word_vectors, use_pretrained, emb_dim, len(input_indexer), embedding_dropout)
        self.encoder = Encoder(emb_dim, hidden_dim, bidirect, n_layers)
        self.decoder = Decoder(emb_dim, hidden_dim, len(self.grammer_indexer))

        self.output_emb = EmbeddingLayer(word_vectors, use_pretrained, emb_dim, len(output_indexer), embedding_dropout)
        self.beam_size = beam_size
        self.decoder_len_limit = decoder_len_limit

    # def forward(self, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor):
    def forward(self, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor, header_length):
        teacher_forcing_value = 0.5
        batch_size = x_tensor.size()[0]

        output_lens = out_lens_tensor.item()

        # (enc_output_each_word, enc_context_mask, enc_final_states_reshaped) = self.encode_input(x_tensor, inp_lens_tensor)
        (enc_output_each_word, enc_context_mask, enc_final_states_reshaped) = self.encode_input(x_tensor,
                                                                                                inp_lens_tensor,
                                                                                                header_length)

        decoder_hidden = enc_final_states_reshaped

        decoder_outputs = [() for i in range(y_tensor.shape[1])]
        type = "V"
        for di in range(y_tensor.shape[1]):
            type = get_type_from_idx(di)

            decoder_input = y_tensor[0, di - 1].unsqueeze(0) if di > 0 else torch.tensor(
                [self.output_indexer.index_of(SOS_SYMBOL)])
            if type == "V":
                decoder_output, decoder_hidden = self.decoder.forward_pred(self.output_emb.forward(decoder_input),
                                                                           decoder_hidden,
                                                                           enc_output_each_word)
            else:
                decoder_output, decoder_hidden = self.decoder.forward_copy(self.output_emb.forward(decoder_input),
                                                                           decoder_hidden,
                                                                           enc_output_each_word)

            decoder_outputs[di] = (decoder_output, type)
            # decoder_outputs[di] = decoder_output
        return decoder_outputs

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        test_derives = []
        with torch.no_grad():
            for test_ex in test_data:
                test_x = test_ex.x_indexed
                enc_output_each_word, enc_context_mask, enc_final_states_reshaped = self.encode_input(
                    torch.tensor([test_x]),
                    torch.tensor(
                        [len(test_x)]),
                    test_ex.header_length)  # TODO: TypeError: encode_input() missing 1 required positional argument: 'header_length'
                decoder_input = torch.tensor([self.output_indexer.index_of(SOS_SYMBOL)])
                decoder_hidden = enc_final_states_reshaped

                end_beam = Beam(self.beam_size)

                beams = [Beam(self.beam_size) for x in range(self.decoder_len_limit + 1)]
                beams[0].add(elt=([], decoder_input, decoder_hidden, False), score=0.0)

                type = "V"

                for di in range(self.decoder_len_limit + 1):
                    type = get_type_from_idx(di)

                    for beam_state, score in beams[di].get_elts_and_scores():

                        y_tokens, decoder_input, decoder_hidden, is_end = beam_state
                        if is_end or di == self.decoder_len_limit:
                            end_beam.add(elt=beam_state, score=score)
                            continue
                        # print(decoder_input)
                        # if len(y_tokens) > 0:
                        #     print(y_tokens[-1])
                        # print("----")
                        output_emb = self.output_emb.forward(decoder_input)
                        if type == "V":
                            decoder_output, decoder_hidden = self.decoder.forward_pred(output_emb, decoder_hidden,
                                                                                       enc_output_each_word)

                        else:  # type=="Q" or type == "C":
                            decoder_output, decoder_hidden = self.decoder.forward_copy(output_emb, decoder_hidden,
                                                                                       enc_output_each_word)
                            if type == "Q":
                                decoder_output[0][0:test_ex.header_length] = 0.

                            decoder_output = log_sum_attn_weight(decoder_output, test_ex.mask)

                        topv, topi = decoder_output.data.topk(self.beam_size)
                        for i in range(self.beam_size):
                            y_tokens_new = y_tokens.copy()
                            if type == "V":
                                token = self.grammer_indexer.get_object(topi[0][i].item())
                            else:
                                token = test_ex.copy_indexer.get_object(topi[0][i].item())
                            prob = topv[0][i].item()
                            if token != "<EOS>" and token != "<GO>":
                                y_tokens_new.append(token)
                            elif token == "<EOS>":
                                is_end = True
                            score_new = score + prob
                            if self.output_indexer.index_of(token) != -1:
                                decoder_input_new = torch.tensor([self.output_indexer.index_of(token)])
                            else:
                                decoder_input_new = torch.tensor([self.output_indexer.index_of(UNK_SYMBOL)])

                            beams[di + 1].add(elt=(y_tokens_new, decoder_input_new, decoder_hidden, is_end),
                                              score=score_new)

                test_ex_de = []
                for beam_state, score in end_beam.get_elts_and_scores():
                    test_ex_de.append(Derivation(test_ex, exp(score), beam_state[0]))
                test_derives.append(test_ex_de)
                # print(test_ex_de[0].y_toks)
        return test_derives

    # def encode_input(self, x_tensor, inp_lens_tensor):
    def encode_input(self, x_tensor, inp_lens_tensor, header_length):
        input_emb = self.input_emb.forward(x_tensor)
        # (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(input_emb, inp_lens_tensor)
        (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(input_emb, inp_lens_tensor,
                                                                                          header_length)
        # enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
        # return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)
        return (enc_output_each_word, enc_context_mask, enc_final_states)


def get_type_from_idx(di):
    if di in [0, 1, 3, 5] or (di >= 6 and di % 4 == 1) or (di >= 6 and di % 4 == 3):
        type = "V"
    elif di in [2, 4] or (di >= 6 and di % 4 == 2):
        type = "C"
    else:  # di>=6 and di%3==2
        type = "Q"
    return type


def log_sum_attn_weight(attn_weights, x_mask):
    # log_attn_weights = log_attn_weights.expand(x_mask.shape[0], -1)

    log_prob = torch.matmul(attn_weights, x_mask)
    log_prob = torch.log(log_prob)
    return log_prob


class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """

    # def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
    def __init__(self, pretrained_word_vectors, use_pretrained, input_dim: int, full_dict_size: int,
                 embedding_dropout_rate: float):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        # self.dropout = nn.Dropout(embedding_dropout_rate)
        if use_pretrained:
            self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_word_vectors.vectors))
        else:
            self.dropout = nn.Dropout(embedding_dropout_rate)
            self.word_embedding = nn.Embedding(full_dict_size, input_dim)
        # self.word_embedding = nn.Embedding.from_pretrained(word_vectors)

    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        embedded_words = self.word_embedding(input)
        # final_embeddings = self.dropout(embedded_words)
        return embedded_words  # final_embeddings


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirect, n_layers):
        super(Encoder, self).__init__()
        self.bidirect = bidirect
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # self.reduce_h_W = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        # self.reduce_h_W = nn.Linear(hidden_dim * 4, hidden_dim, bias=True)
        self.layer1_reduce_h_W = nn.Linear(hidden_dim * 4, hidden_dim, bias=True)  # hidden_dim=100
        self.layer2_reduce_h_W = nn.Linear(hidden_dim * 4, hidden_dim, bias=True)
        self.layer3_reduce_h_W = nn.Linear(hidden_dim * 4, hidden_dim, bias=True)

        self.layer1_reduce_c_W = nn.Linear(hidden_dim * 4, hidden_dim, bias=True)  # hidden_dim=100
        self.layer2_reduce_c_W = nn.Linear(hidden_dim * 4, hidden_dim, bias=True)
        self.layer3_reduce_c_W = nn.Linear(hidden_dim * 4, hidden_dim, bias=True)

        self.reduce_c_W = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers,
                           batch_first=True, dropout=0,
                           bidirectional=self.bidirect)
        self.init_weight()
        # pass n_layers=3

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
        return torch.from_numpy(np.asarray(
            [[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, embedded_words, input_lens, header_length):
        """
        Runs the forward pass of the LSTM
        :param embedded_words: [batch size x sent len x input dim] tensor
        :param input_lens: [batch size]-length vector containing the length of each input sentence
        :return: output (each word's representation), context_mask (a mask of 0s and 1s
        reflecting where the model's output should be considered), and h_t, a *tuple* containing
        the final states h and c from the encoder for each sentence.
        """
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        embedded_words_part1 = embedded_words[:, 0:header_length, :]
        embedded_words_part2 = embedded_words[:, header_length:, :]

        # packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True, enforce_sorted=False)
        packed_embedding_part1 = nn.utils.rnn.pack_padded_sequence(embedded_words_part1, torch.tensor([header_length]),
                                                                   batch_first=True, enforce_sorted=False)
        packed_embedding_part2 = nn.utils.rnn.pack_padded_sequence(embedded_words_part2,
                                                                   input_lens - torch.tensor([header_length]),
                                                                   batch_first=True, enforce_sorted=False)

        output_part1, hn_part1 = self.rnn(packed_embedding_part1)
        output_part2, hn_part2 = self.rnn(packed_embedding_part2, hn_part1)

        # output = torch.cat((output_part1, output_part2))

        # output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        output_part1, sent_lens_part1 = nn.utils.rnn.pad_packed_sequence(output_part1)
        output_part2, sent_lens_part2 = nn.utils.rnn.pad_packed_sequence(output_part2)
        output = torch.cat((output_part1, output_part2))

        sent_lens = sent_lens_part1 + sent_lens_part2
        # output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors

        max_length = input_lens.max().item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # header_length = 6
        # hid_state_header = output[header_length]
        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h_part1, c_part1 = hn_part1[0], hn_part1[1]
            h_part2, c_part2 = hn_part2[0], hn_part2[1]

            h_part1 = h_part1.view(self.n_layers, 2, 1,
                                   self.hidden_dim)  # h_n.view(num_layers, num_directions, batch, hidden_size)
            h_part2 = h_part2.view(self.n_layers, 2, 1, self.hidden_dim)

            c_part1 = c_part1.view(self.n_layers, 2, 1,
                                   self.hidden_dim)  # h_n.view(num_layers, num_directions, batch, hidden_size)
            c_part2 = c_part2.view(self.n_layers, 2, 1, self.hidden_dim)
            # h_part1[0,0,:,:].size()
            # h, c = hn[0], hn[1]  # h & c = (num_layers * num_directions, batch, hidden_size)
            # Grab the representations from forward and backward LSTMs
            # h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]),
            #                                                    dim=1)  # h[0] = [batch, hidden_size], h_ & c_ = [batch, hidden_size*num_directions]

            # h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # layer1_h_ = torch.cat((h_part1[0], h_part1[1], h_part2[0], h_part2[1]), dim=1) # h_part1[0].size()=torch.Size([1, 100])
            layer1_h_ = torch.cat((h_part1[0, 0, :, :], h_part1[0, 1, :, :], h_part2[0, 0, :, :], h_part2[0, 1, :, :]),
                                  dim=1)
            layer2_h_ = torch.cat((h_part1[1, 0, :, :], h_part1[1, 1, :, :], h_part2[1, 0, :, :], h_part2[1, 1, :, :]),
                                  dim=1)
            layer3_h_ = torch.cat((h_part1[2, 0, :, :], h_part1[2, 1, :, :], h_part2[2, 0, :, :], h_part2[2, 1, :, :]),
                                  dim=1)

            layer1_c_ = torch.cat((c_part1[0, 0, :, :], c_part1[0, 1, :, :], c_part2[0, 0, :, :], c_part2[0, 1, :, :]),
                                  dim=1)
            layer2_c_ = torch.cat((c_part1[1, 0, :, :], c_part1[1, 1, :, :], c_part2[1, 0, :, :], c_part2[1, 1, :, :]),
                                  dim=1)
            layer3_c_ = torch.cat((c_part1[2, 0, :, :], c_part1[2, 1, :, :], c_part2[2, 0, :, :], c_part2[2, 1, :, :]),
                                  dim=1)

            # h_ = torch.cat((h[0], h[1], hid_state_header), dim=1)
            # c_ = torch.cat((c[0], c[1]), dim=1)

            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            # new_h = self.reduce_h_W(h_)  # new_h = [batch, hidden_size]
            # new_c = self.reduce_c_W(c_)
            # h_t = (new_h, new_c)

            layer1_new_h = self.layer1_reduce_h_W(layer1_h_)  # new_h = [batch, hidden_size]
            layer2_new_h = self.layer2_reduce_h_W(layer2_h_)
            layer3_new_h = self.layer3_reduce_h_W(layer3_h_)
            all_new_h = torch.stack((layer1_new_h, layer2_new_h, layer3_new_h))

            layer1_new_c = self.layer1_reduce_c_W(layer1_c_)  # new_h = [batch, hidden_size]
            layer2_new_c = self.layer2_reduce_c_W(layer2_c_)
            layer3_new_c = self.layer3_reduce_c_W(layer3_c_)
            all_new_c = torch.stack((layer1_new_c, layer2_new_c, layer3_new_c))

            all_h_t = (all_new_h, all_new_c)
            # new_c = self.reduce_c_W(c_)
            # h_t = (new_h, new_c)

        # else:
        #     h, c = hn[0][0], hn[1][0]
        #     h_t = (h, c)
        # return (output, context_mask, h_t)
        return (output, context_mask, all_h_t)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden, encoder_outputs):
        # values = self.linear(encoder_outputs.squeeze(0)).unsqueeze(0)
        # attn_logits = hidden.bmm(values.transpose(1, 2))
        values = self.linear(encoder_outputs.squeeze(0))
        attn_logits = hidden.bmm(values.transpose(0, 1).transpose(1, 2))

        attn_weights = F.softmax(attn_logits, dim=-1)
        # context = attn_weights.bmm(values)
        context = attn_weights.bmm(values.transpose(0, 1))

        return attn_weights, context


# values2 = self.linear(encoder_outputs.squeeze(0))
# values3 = values2.transpose(1, 2)
# temp = values.transpose(1, 2)
# temp = values2.transpose(0, 1).transpose(1, 2)
class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        :param input_size: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        """
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=3, dropout=0.)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.attention = Attention(self.hidden_size)
        self.init_weight()

    def init_weight(self):
        """
        Initializes weight matrices using Xavier initialization
        :return:
        """
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        # nn.init.xavier_uniform_(self.out.weight)
        # nn.init.constant_(self.out.bias, 0)

    def forward_pred(self, embedded_words, hidden, encoder_outputs):
        output, hidden = self.rnn(embedded_words.view(len(embedded_words), 1, -1), hidden)
        # embedded_words.view(len(embedded_words), 1, -1).size()=torch.Size([1, 1, 200])
        # hidden
        attn_weights, context = self.attention(output, encoder_outputs)
        output = output.view(len(embedded_words), -1)
        context = context.view(len(embedded_words), -1)
        output = self.log_softmax(self.out(torch.cat((output, context), dim=1)))
        return output, hidden

    def forward_copy(self, embedded_words, hidden, encoder_outputs):
        output, hidden = self.rnn(embedded_words.view(len(embedded_words), 1, -1), hidden)
        attn_weights, context = self.attention(output, encoder_outputs)
        return attn_weights.view(1, -1), hidden


def train_model(word_vectors, train_data: List[Example], dev_data: List[Example], input_indexer, output_indexer, args):
    # AttributeError: 'tuple' object has no attribute 'words'
    # all_train_input_data
    # all_test_input_data

    # all_train_output_data
    # all_test_output_data
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))  # input_max_len = 19
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False)
    all_test_input_data = make_padded_input_tensor(dev_data, input_indexer, input_max_len, reverse_input=False)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(dev_data, output_indexer, output_max_len)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # # Format data
    # all_input_lens = torch.from_numpy(np.asarray([len(ex.x_indexed) for ex in train_data]))
    # all_train_input_data = torch.from_numpy(all_train_input_data)
    # all_test_input_data = torch.from_numpy(all_test_input_data)

    # all_output_lens = torch.from_numpy(np.asarray([len(ex.y_indexed) for ex in train_data]))
    # all_train_output_data = torch.from_numpy(all_train_output_data)
    # all_test_output_data = torch.from_numpy(all_test_output_data)

    seq2seq = Seq2Seq(use_pretrained=args.use_pretrained,
                      word_vectors=word_vectors,
                      input_indexer=input_indexer,
                      output_indexer=output_indexer,
                      emb_dim=args.emb_dim,
                      hidden_dim=args.hidden_dim,
                      embedding_dropout=args.embedding_dropout)

    n_epochs = args.epochs
    n_exs = all_train_input_data.shape[0]
    # n_exs = 10
    lr = args.lr
    batch_size = args.batch_size

    optimizer = optim.Adagrad(seq2seq.parameters(), lr)
    # criterion = nn.CrossEntropyLoss(ignore_index = word_vectors.word_indexer.index_of('PAD'))
    criterion = nn.NLLLoss()

    seq2seq.train()
    # with autograd.detect_anomaly():
    for epoch_idx in range(n_epochs):
        start = time.time()
        
        ex_indices = [i for i in range(0, n_exs)]
        np.random.shuffle(ex_indices)
        total_loss = 0.0
        for ex_idx in ex_indices:
            sample = train_data[ex_idx]
            header_length = sample.header_length
            # sample = all_train_input_data[ex_idx]
            x_tensor = torch.tensor(sample.x_indexed).unsqueeze(0)
            inp_lens_tensor = torch.tensor([len(sample.x_indexed)])
            y_tensor = torch.tensor(sample.y_indexed).unsqueeze(0)
            out_lens_tensor = torch.tensor([len(sample.y_indexed)])
            for i in range(y_tensor.shape[1]):
                type = get_type_from_idx(i)
                if type == "V":
                    y_tensor[0][i] = seq2seq.grammer_indexer.index_of(
                        output_indexer.get_object(y_tensor[0][i].item()))
                else:
                    y_tensor[0][i] = sample.copy_indexer.index_of(
                        output_indexer.get_object(y_tensor[0][i].item()))

            optimizer.zero_grad()
            # decoder_outputs = seq2seq.forward(x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor)
            decoder_outputs = seq2seq.forward(x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor, header_length)
            loss = 0.0
            # y_tensor_ = y_tensor.clone()
            for idx, (output, type) in enumerate(decoder_outputs):
                # if type == "V":
                #     y_tensor_[0][idx] = seq2seq.grammer_indexer.index_of(
                #         output_indexer.get_object(y_tensor[0][idx].item()))
                # loss += criterion(output, y_tensor_[:, idx])
                if type != "V":
                    output = log_sum_attn_weight(output, sample.mask)
                    # attn_over_tok = torch.zeros((1, len(sample.copy_indexer)))
                    # for tok in sample.tok_to_idx:
                    #     idx_tok_to_idx = sample.tok_to_idx[tok]
                    #     attn_over_tok[0][sample.copy_indexer.index_of(tok)] = attn[0][idx_tok_to_idx]
                    # output = attn_over_tok
                    # y_tensor_[0][idx] = sample.copy_indexer.index_of(
                    #     output_indexer.get_object(y_tensor[0][idx].item()))
                if idx == -1:
                    print(idx)
                loss += criterion(output, y_tensor[:, idx])

            total_loss += loss

            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch_idx + 1, total_loss))
        print("Time elapsed: ", time.time() - start)
    return seq2seq


def createGrammerIndexer():
    indexer = Indexer()
    indexer.add_and_get_index("select")
    indexer.add_and_get_index("from")
    indexer.add_and_get_index("where")
    indexer.add_and_get_index("id")
    indexer.add_and_get_index("max")
    indexer.add_and_get_index("min")
    indexer.add_and_get_index("count")
    indexer.add_and_get_index("sum")
    indexer.add_and_get_index("avg")
    indexer.add_and_get_index("and")
    indexer.add_and_get_index("=")
    indexer.add_and_get_index(">")
    indexer.add_and_get_index(">=")
    indexer.add_and_get_index("<")
    indexer.add_and_get_index("<=")
    indexer.add_and_get_index(EOS_SYMBOL)
    indexer.add_and_get_index(GO_SYMBOL)
    return indexer
