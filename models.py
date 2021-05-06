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
    """
    Command-line arguments to the system related to your model.  Feel free to extend here.  
    """
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')
    parser.add_argument('--emb_dim', type=int, default=100, help='word embedding dimensions')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden layer size')
    # parser.add_argument('--encoder_len_limit', type=int, default=19, help='input length limit of the encoder')
    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # N.B. a list!
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


###################################################################################################################
# You do not have to use any of the classes in this file, but they're meant to give you a starting implementation.
# for your network.
###################################################################################################################

class Seq2SeqSemanticParser(nn.Module):
    def __init__(self, input_indexer, output_indexer, emb_dim, hidden_size, embedding_dropout=0.2, bidirect=True):
        # We've include some args for setting up the input embedding and encoder
        # You'll need to add code for output embedding and decoder
        super(Seq2SeqSemanticParser, self).__init__()
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        
        self.input_emb = EmbeddingLayer(emb_dim, len(input_indexer), embedding_dropout)
        self.encoder = RNNEncoder(emb_dim, hidden_size, bidirect)
        # input_emb_decoder=output_emb
        self.output_emb = EmbeddingLayer(emb_dim, len(output_indexer), embedding_dropout)
        self.decoder = RNNDecoder(emb_dim, hidden_size, len(output_indexer))
        
        # self.args = args
        # raise Exception("implement me!")


    def forward(self, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor):
        """
        

        Parameters
        ----------
        x_tensor : tensor
            [batch_size, sen_len_padded=19]
        inp_lens_tensor : tensor
            [batch_size]
        y_tensor : tensor
            [batch_size, sen_len_padded=65]
        out_lens_tensor : tensor
            [batch_size]

        Returns
        -------
        fc_outputs : TYPE
            DESCRIPTION.

        """
        """
        :param x_tensor/y_tensor: either a non-batched input/output [sent len x voc size] or a batched input/output
        [batch size x sent len x voc size]
        
        x_tensor: [sent len]
        :param inp_lens_tensor/out_lens_tensor: either a vecor of input/output length [batch size] or a single integer.
        lengths aren't needed if you don't batchify the training.
        :return: loss of the batch
        """
        teacher_forcing_value = 0.5
        batch_size = x_tensor.size()[0] 
        
        output_lens = out_lens_tensor.item()

        (enc_output_each_word, enc_context_mask, enc_final_states_reshaped) = self.encode_input(x_tensor, inp_lens_tensor)
        # enc_output_each_word.size()=[sent len, batch_size=1, num_directions*hidden_size]
        # enc_context_mask.size() = [batch_size, sent len]
        # enc_final_states_reshaped[0].size()=(num_layers * num_directions=1, batch=1, hidden_size)
        
        h_t = enc_final_states_reshaped
        # store results of fc_output
        fc_outputs = torch.zeros(output_lens, batch_size, len(self.output_indexer))
        #fc_outputs.size() = [output_lens, batch_size=1, output_vocab_size=153]
        
        DECODER_START = torch.as_tensor(self.output_indexer.index_of('<SOS>'))
        embedded_words_decoder = self.output_emb.forward(DECODER_START) # embedded_words = [batch_size, sent len = 19, emb_dim]
        
        # make dimensions right
        embedded_words_decoder = embedded_words_decoder.unsqueeze(0).unsqueeze(0) 
        # embedded_words_decoder.size()=(1,1,emb_dim)

        for output_idx in range(output_lens):
            (fc_output, h_t) = self.decoder.forward(embedded_words_decoder, h_t) # h_t[0].size()=(num_layers * num_directions=1, batch=1, hidden_size)
            fc_outputs[output_idx] = fc_output
            predicted_token_index = fc_output.argmax(1) 
            
            if np.random.random() < teacher_forcing_value:
                embedded_words_decoder = self.output_emb.forward(y_tensor[0:batch_size, output_idx]) # embedded_words_decoder.size()=torch.Size([1, hid_size])
            else:
                embedded_words_decoder = self.output_emb.forward(predicted_token_index) #predicted_token.size()=torch.Size([1])
            embedded_words_decoder = embedded_words_decoder.unsqueeze(0)
            
        return fc_outputs
            
        # raise Exception("implement me!")

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        

        Parameters
        ----------
        test_data : List[Example]
            dev_data_indexed
            dev_data_indexed[1].x_indexed
            >>> [38, 121, 9, 186, 187, 8]
            dev_data_indexed[1].y_indexed
        Returns
        -------
        List[List[Derivation]]
            DESCRIPTION.

        """
        self.eval()
        # input_max_len = 19#self.args.encoder_len_limit
        input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
        
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, reverse_input=False) # array=(n_dev_exs,input_max_len=19)
        
        output_max_len = 65#self.args.decoder_len_limit
        all_test_output_data = make_padded_output_tensor(test_data, self.output_indexer, output_max_len) # array=(n_dev_exs,input_max_len=65)
        
        # format data
        all_input_lens = torch.from_numpy(np.asarray([len(ex.x_indexed) for ex in test_data]))
        all_test_input_data = torch.from_numpy(all_test_input_data)
        all_test_output_data = torch.from_numpy(all_test_output_data)
        n_exs = all_test_input_data.size()[0]
        Derivation_list = []
        for ex_idx in range(n_exs):
            x_tensor = all_test_input_data[ex_idx].unsqueeze(0)
            inp_lens_tensor = all_input_lens[ex_idx].unsqueeze(0)
            (enc_output_each_word, enc_context_mask, enc_final_states_reshaped) = self.encode_input(x_tensor, inp_lens_tensor)
            h_t = enc_final_states_reshaped
            
            DECODER_START = torch.as_tensor(self.output_indexer.index_of('<SOS>'))
            embedded_words_decoder = self.output_emb.forward(DECODER_START)
            embedded_words_decoder = embedded_words_decoder.unsqueeze(0).unsqueeze(0) 
            
            output_idx = 0
            predicted_token = '<SOS>'
            Derivation_token = [] #Derivation_token = ['<PAD>'] * output_max_len
            for output_idx in range(output_max_len):
                (fc_output, h_t) = self.decoder.forward(embedded_words_decoder, h_t)
                predicted_token_index = fc_output.argmax(1)
                predicted_token = self.output_indexer.get_object(predicted_token_index.item())
                
                if predicted_token != '<EOS>':
                    Derivation_token.append(predicted_token)#Derivation_token[output_idx] = predicted_token
                    embedded_words_decoder = self.output_emb.forward(predicted_token_index)
                    embedded_words_decoder = embedded_words_decoder.unsqueeze(0)
                else:
                    break
    
            Derivation_list.append([Derivation(test_data[ex_idx], 1.0, Derivation_token)])
            # print(Derivation_token)
        return Derivation_list

    
    def encode_input(self, x_tensor, inp_lens_tensor):
        """
        Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
        inp_lens_tensor lengths.
        YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
        as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
        :param x_tensor: [batch size, sent len] tensor of input token indices
        :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
        :param model_input_emb: EmbeddingLayer
        :param model_enc: RNNEncoder
        :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
        are real and which ones are pad tokens), and the encoder final states (h and c tuple)
        E.g., calling this with x_tensor (0 is pad token):
        [[12, 25, 0, 0],
        [1, 2, 3, 0],
        [2, 0, 0, 0]]
        inp_lens = [2, 3, 1]
        will return outputs with the following shape:
        enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
        enc_final_states = 3 x dim
        """
        input_emb = self.input_emb.forward(x_tensor)
        (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(input_emb, inp_lens_tensor)
        enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
        return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


class Seq2SeqSemanticParserAttention(nn.Module):
    def __init__(self, input_indexer, output_indexer, emb_dim, hidden_size, embedding_dropout=0.2, bidirect=True):
        # We've include some args for setting up the input embedding and encoder
        # You'll need to add code for output embedding and decoder
        super(Seq2SeqSemanticParserAttention, self).__init__()
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        
        self.input_emb = EmbeddingLayer(emb_dim, len(input_indexer), embedding_dropout)
        self.encoder = RNNEncoder(emb_dim, hidden_size, bidirect)
        # input_emb_decoder=output_emb
        self.output_emb = EmbeddingLayer(emb_dim, len(output_indexer), embedding_dropout)
        self.decoder = RNNDecoderAttention(emb_dim, hidden_size, len(output_indexer))
        
        # self.args = args
        # raise Exception("implement me!")


    def forward(self, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor):
        """
        
        Parameters
        ----------
        x_tensor : tensor
            [batch_size, sen_len_padded=19]
        inp_lens_tensor : tensor
            [batch_size]
        y_tensor : tensor
            [batch_size, sen_len_padded=65]
        out_lens_tensor : tensor
            [batch_size]
        Returns
        -------
        fc_outputs : TYPE
            DESCRIPTION.
        """
        """
        :param x_tensor/y_tensor: either a non-batched input/output [sent len x voc size] or a batched input/output
        [batch size x sent len x voc size]
        
        x_tensor: [sent len]
        :param inp_lens_tensor/out_lens_tensor: either a vecor of input/output length [batch size] or a single integer.
        lengths aren't needed if you don't batchify the training.
        :return: loss of the batch
        """
        teacher_forcing_value = 0.5
        batch_size = x_tensor.size()[0] 
        
        output_lens = out_lens_tensor.item()

        (enc_output_each_word, enc_context_mask, enc_final_states_reshaped) = self.encode_input(x_tensor, inp_lens_tensor)
        # enc_output_each_word.size()=[sent len, batch_size=1, num_directions*hidden_size]
        # enc_context_mask.size() = [batch_size, sent len]
        # enc_final_states_reshaped[0].size()=(num_layers * num_directions=1, batch=1, hidden_size)
        # enc_output_each_word.size()
        h_t = enc_final_states_reshaped
        h_t_enc_all = enc_output_each_word #h_t_all[3].size()
        # store results of fc_output
        fc_outputs = torch.zeros(output_lens, batch_size, len(self.output_indexer))
        
        DECODER_START = torch.as_tensor(self.output_indexer.index_of('<SOS>'))
        embedded_words_decoder = self.output_emb.forward(DECODER_START) # embedded_words = [batch_size, sent len = 19, emb_dim]
        
        # make dimensions right
        embedded_words_decoder = embedded_words_decoder.unsqueeze(0).unsqueeze(0) 
        # embedded_words_decoder.size()=(1,1,emb_dim)
        
        for output_idx in range(output_lens):
            (fc_output, h_t) = self.decoder.forward(embedded_words_decoder, h_t, h_t_enc_all)
            fc_outputs[output_idx] = fc_output
            predicted_token_index = fc_output.argmax(1)

            if np.random.random() < teacher_forcing_value:
                embedded_words_decoder = self.output_emb.forward(y_tensor[0:batch_size, output_idx]) # embedded_words_decoder.size()=torch.Size([1, hid_size])
            else:
                embedded_words_decoder = self.output_emb.forward(predicted_token_index) #predicted_token.size()=torch.Size([1])
            embedded_words_decoder = embedded_words_decoder.unsqueeze(0)
        return fc_outputs


    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        
        Parameters
        ----------
        test_data : List[Example]
            dev_data_indexed
            dev_data_indexed[1].x_indexed
            >>> [38, 121, 9, 186, 187, 8]
            dev_data_indexed[1].y_indexed
        Returns
        -------
        List[List[Derivation]]
            DESCRIPTION.
        """
        self.eval()
        # input_max_len = 19#self.args.encoder_len_limit
        input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
        
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, reverse_input=False) # array=(n_dev_exs,input_max_len=19)
        
        output_max_len = 65#self.args.decoder_len_limit
        all_test_output_data = make_padded_output_tensor(test_data, self.output_indexer, output_max_len) # array=(n_dev_exs,input_max_len=65)
        
        # format data
        all_input_lens = torch.from_numpy(np.asarray([len(ex.x_indexed) for ex in test_data]))
        all_test_input_data = torch.from_numpy(all_test_input_data)
        all_test_output_data = torch.from_numpy(all_test_output_data)
        n_exs = all_test_input_data.size()[0]
        Derivation_list = []
        for ex_idx in range(n_exs):
            x_tensor = all_test_input_data[ex_idx].unsqueeze(0)
            inp_lens_tensor = all_input_lens[ex_idx].unsqueeze(0)
            (enc_output_each_word, enc_context_mask, enc_final_states_reshaped) = self.encode_input(x_tensor, inp_lens_tensor)
            h_t = enc_final_states_reshaped
            h_t_enc_all = enc_output_each_word
            
            DECODER_START = torch.as_tensor(self.output_indexer.index_of('<SOS>'))
            embedded_words_decoder = self.output_emb.forward(DECODER_START)
            embedded_words_decoder = embedded_words_decoder.unsqueeze(0).unsqueeze(0) 
            
            # output_idx = 0
            predicted_token = '<SOS>'
            Derivation_token = [] #Derivation_token = ['<PAD>'] * output_max_len
            for output_idx in range(output_max_len):
                (fc_output, h_t) = self.decoder.forward(embedded_words_decoder, h_t, h_t_enc_all)
                predicted_token_index = fc_output.argmax(1)
                predicted_token = self.output_indexer.get_object(predicted_token_index.item())
                
                if predicted_token != '<EOS>':
                    Derivation_token.append(predicted_token)#Derivation_token[output_idx] = predicted_token
                    embedded_words_decoder = self.output_emb.forward(predicted_token_index)
                    embedded_words_decoder = embedded_words_decoder.unsqueeze(0)
                else:
                    break
    
            Derivation_list.append([Derivation(test_data[ex_idx], 1.0, Derivation_token)])
            # print(Derivation_token)
        return Derivation_list

    
    def encode_input(self, x_tensor, inp_lens_tensor):
        """
        Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
        inp_lens_tensor lengths.
        YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
        as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
        :param x_tensor: [batch size, sent len] tensor of input token indices
        :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
        :param model_input_emb: EmbeddingLayer
        :param model_enc: RNNEncoder
        :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
        are real and which ones are pad tokens), and the encoder final states (h and c tuple)
        E.g., calling this with x_tensor (0 is pad token):
        [[12, 25, 0, 0],
        [1, 2, 3, 0],
        [2, 0, 0, 0]]
        inp_lens = [2, 3, 1]
        will return outputs with the following shape:
        enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
        enc_final_states = 3 x dim
        """
        input_emb = self.input_emb.forward(x_tensor)
        (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(input_emb, inp_lens_tensor)
        enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
        return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """
    def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


class RNNEncoder(nn.Module):
    """
    One-layer RNN encoder for batched inputs -- handles multiple sentences at once. To use in non-batched mode, call it
    with a leading dimension of 1 (i.e., use batch size 1)
    """
    def __init__(self, input_size: int, hidden_size: int, bidirect: bool):
        """
        :param input_size: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                               dropout=0., bidirectional=self.bidirect)
        self.init_weight()

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
###################################################################################################################
# End optional classes
###################################################################################################################


class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = output_size
        self.rnn = nn.LSTM(input_size, hidden_size, dropout=0) #input_size
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, embedded_words, h_t):
        output, h_t_decoder = self.rnn(embedded_words, h_t) 
        # output.size() = (seq_len=1, batch=1, hidden_size)
        # h_t_decoder[0].size() = (num_layers * num_directions=1, batch=1, hidden_size)
        # output = [sent len=1, batch_size=1, num_directions(1)*hidden_size],  hn = (h, c)
        fc_output = self.fc(output.squeeze(0)) # fc_output.size() = [batch, output_size=output_vocab_size=153]
        return (fc_output, h_t_decoder)
    

class RNNDecoderAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNDecoderAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = output_size
        self.rnn = nn.LSTM(input_size, hidden_size, dropout=0) #input_size
        self.fc = nn.Linear(hidden_size + hidden_size*2, output_size) #TODO: *2 bidirectional
        self.V = nn.Linear(hidden_size, hidden_size*2) # use another weight matrix-V to multiply #TODO: *2 bidirectional
        nn.init.xavier_uniform_(self.V.weight)
        
    def forward(self, embedded_words, h_t, h_t_enc_all):
        
        output, h_t_decoder = self.rnn(embedded_words, h_t)
        # output.size() = (seq_len=1, batch=1, hidden_size)
        # h_t_decoder[0].size() = (num_layers * num_directions=1, batch=1, hidden_size)
        # output.size() = [sent len=1, batch_size=1, num_directions(1)*hidden_size],  hn = (h, c)
        # h_t_enc_all.size() = torch.Size([input_len, batch, num_directions(2)*hidden_size])

        input_len = h_t_enc_all.size()[0]
        h_n = h_t_decoder[0].squeeze(0) # h_n.size()=torch.Size([batch=1, hidden_size])
        batch_size = h_n.size()[0]
        
        
        e_weights = torch.zeros(batch_size, input_len)
        for input_idx in range(input_len):
            for batch_idx in range(batch_size):
                h_t_enc_T = h_t_enc_all[input_idx].transpose(0, 1) # h_t_enc_T.size() = torch.Size([num_directions(2)*hidden_size, batch])
                e_weight = torch.dot(self.V(h_n)[batch_idx], h_t_enc_T[:, batch_idx])
                e_weights[batch_idx, input_idx] = e_weight
        
        alpha = torch.nn.functional.softmax(e_weights, dim=1) # alpha.size()=[batch, input_len]
        alpha = alpha.unsqueeze(1) # make dimensions right to use torch.bmm
        c_vector = torch.bmm(alpha, h_t_enc_all.view(batch_size, input_len, -1)).squeeze(1)
        # h_t_enc_all.view(batch_size, input_len, -1) # h_t_enc_all.size()
        
        output = torch.cat((c_vector, h_n), 1)
        fc_output = self.fc(output)
        return (fc_output, h_t_decoder)
    # c_vector[batch_idx, ] = torch.dot(alpha[batch_idx, ] * h_t_enc_all.view(batch_size, input_len, -1)[batch_idx, ])
    # alpha[batch_idx, ].size() alpha1 = alpha.unsqueeze(1)#.size() #alpha.size()
    # torch.bmm(alpha1, h_t_enc_all.view(batch_size, input_len, -1)).size()
def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])

def train_model(train_data: List[Example], dev_data: List[Example], input_indexer, output_indexer, args):
    #  -> Seq2SeqSemanticParser
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param dev_data: Development set in case you wish to evaluate during training
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data])) # input_max_len = 19
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False)
    all_test_input_data = make_padded_input_tensor(dev_data, input_indexer, input_max_len, reverse_input=False)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(dev_data, output_indexer, output_max_len)

    # if args.print_dataset:
    #     print("Train length: %i" % input_max_len)
    #     print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    #     print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))
        
    # Format data
    all_input_lens = torch.from_numpy(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = torch.from_numpy(all_train_input_data)
    all_test_input_data = torch.from_numpy(all_test_input_data)
    
    all_output_lens = torch.from_numpy(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = torch.from_numpy(all_train_output_data)
    all_test_output_data = torch.from_numpy(all_test_output_data)
    
    # First create a model
    # seq2seq = Seq2SeqSemanticParser(input_indexer=input_indexer,
    #                                 output_indexer=output_indexer,
    #                                 emb_dim=args.emb_dim,
    #                                 hidden_size=args.hidden_size)
    
    # attention model
    seq2seq = Seq2SeqSemanticParserAttention(input_indexer=input_indexer,
                                             output_indexer=output_indexer,
                                             emb_dim=args.emb_dim,
                                             hidden_size=args.hidden_size)
    
    n_epochs = args.epochs
    n_exs = all_train_input_data.size()[0] # number of training examples
    # n_exs = 10
    lr = args.lr
    batch_size = args.batch_size
    
    optimizer = optim.Adam(seq2seq.parameters(), lr)
    criterion = nn.CrossEntropyLoss(ignore_index = output_indexer.index_of('<PAD>'))
    # Then loop over epochs, loop over examples, and given some indexed words
    # call your seq-to-seq model, accumulate losses, update parameters    
    seq2seq.train()
    for epoch_idx in range(n_epochs):
        ex_indices = [i for i in range(0, n_exs)]
        np.random.shuffle(ex_indices)
        total_loss = 0.0
        for ex_idx in ex_indices:
             x_tensor = all_train_input_data[ex_idx:(ex_idx+batch_size)] # # exs = [sent len = 19]
             inp_lens_tensor = all_input_lens[ex_idx:(ex_idx+batch_size)] # input_lens = [batch_size]
             
             y_tensor = all_train_output_data[ex_idx:(ex_idx+batch_size)]
             # y_tensor = all_train_input_data[ex_idx:(ex_idx+batch_size)]
             out_lens_tensor = all_output_lens[ex_idx:(ex_idx+batch_size)]
             # out_lens_tensor = all_input_lens[ex_idx:(ex_idx+batch_size)]
             
             optimizer.zero_grad()
             fc_outputs = seq2seq.forward(x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor) # fc_outputs.size()=[out_lens, batch, output_vocab_size]
             outputs = fc_outputs.view(-1, len(output_indexer)) # outputs.size()=[out_lens*batch, output_vocab_size=n_classes]
             target = (y_tensor.view(-1)[:out_lens_tensor]).type(torch.LongTensor)
             
             loss = criterion(outputs, target)
             total_loss += loss

             loss.backward()
             optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch_idx + 1, total_loss))
    return seq2seq
    #raise Exception("Implement the rest of me to train your encoder-decoder model")
    
