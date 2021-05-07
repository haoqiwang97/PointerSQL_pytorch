from utils import *
from typing import List, Tuple
from collections import Counter
import numpy as np
import torch


class Example(object):
    """
    Wrapper class for a single (natural language, logical form) input/output (x/y) pair
    Attributes:
        x: the natural language as one string
        x_tok: tokenized natural language as a list of strings
        x_indexed: indexed tokens, a list of ints
        y: the raw logical form as a string
        y_tok: tokenized logical form, a list of strings
        y_indexed: indexed logical form, a list of ints
    """

    def __init__(self, x: str, x_tok: List[str], x_indexed: List[int], y, y_tok, y_indexed, header_length):
        self.x = x
        self.x_tok = x_tok
        self.x_indexed = x_indexed
        self.y = y
        self.y_tok = y_tok
        self.y_indexed = y_indexed
        self.header_length = header_length
        tok_to_idx = {}
        for idx, tok in enumerate(x_tok):
            if tok not in tok_to_idx:
                tok_to_idx[tok] = []
            tok_to_idx[tok].append(idx)
        self.copy_indexer = Indexer()
        for tok in tok_to_idx.keys():
            self.copy_indexer.add_and_get_index(tok)
        self.mask = torch.zeros((len(x_tok), len(tok_to_idx)))
        for tok in tok_to_idx:
            idx = tok_to_idx[tok]
            for i in idx:
                self.mask[i][self.copy_indexer.index_of(tok)] = 1.

    def __repr__(self):
        return " ".join(self.x_tok) + " => " + " ".join(self.y_tok) + "\n   indexed as: " + repr(
            self.x_indexed) + " => " + repr(self.y_indexed)

    def __str__(self):
        return self.__repr__()


# class Example(object):
#     """
#     Wrapper class for a single (natural language, logical form) input/output (x/y) pair
#     Attributes:
#         x: the natural language as one string
#         x_tok: tokenized natural language as a list of strings
#         x_indexed: indexed tokens, a list of ints
#         y: the raw logical form as a string
#         y_tok: tokenized logical form, a list of strings
#         y_indexed: indexed logical form, a list of ints
#     """
#     def __init__(self, x: str, x_tok: List[str], x_indexed: List[int], y, y_tok, y_indexed):
#         self.x = x
#         self.x_tok = x_tok
#         self.x_indexed = x_indexed
#         self.y = y
#         self.y_tok = y_tok
#         self.y_indexed = y_indexed

#     def __repr__(self):
#         return " ".join(self.x_tok) + " => " + " ".join(self.y_tok) + "\n   indexed as: " + repr(self.x_indexed) + " => " + repr(self.y_indexed)

#     def __str__(self):
#         return self.__repr__()


class Derivation(object):
    """
    Wrapper for a possible solution returned by the model associated with an Example. Note that y_toks here is a
    predicted y_toks, and the Example itself contains the gold y_toks.
    Attributes:
          example: The underlying Example we're predicting on
          p: the probability associated with this prediction
          y_toks: the tokenized output prediction
    """

    def __init__(self, example: Example, p, y_toks):
        self.example = example
        self.p = p
        self.y_toks = y_toks

    def __str__(self):
        return "%s (%s)" % (self.y_toks, self.p)

    def __repr__(self):
        return self.__str__()


PAD_SYMBOL = "<PAD>"
UNK_SYMBOL = "<UNK>"
SOS_SYMBOL = "<SOS>"
EOS_SYMBOL = "<EOS>"
GO_SYMBOL = "<GO>"


def load_datasets(train_path: str, dev_path: str, test_path: str) -> (
        List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]):
    train_raw = load_dataset(train_path)
    dev_raw = load_dataset(dev_path)
    test_raw = load_dataset(test_path)
    return train_raw, dev_raw, test_raw


def load_dataset(filename: str) -> List[Tuple[str, str, int]]:
    dataset = []
    with open(filename) as f:
        raw_lines = f.readlines()
        n_lines = len(raw_lines)
        # each example takes 4 lines
        # 1st, table and column names, eg, raw_lines[0]='1-10015132-11 player no. nationality position years^in^toronto school/club^team\n'
        # 2nd, query, eg, raw_lines[1]='what position does the player who played for school/club^team butler^cc^(ks) play ?\n'
        # 3rd, y, ground truth, eg, raw_lines[2]='1-10015132-11 select position school/club^team = butler^cc^(ks)\n'
        # 4th, '\n', empty, eg, raw_lines[3]='\n'
        # c=Counter()#
        n_examples = n_lines // 4
        for example_idx in range(n_examples):
            table_column_name = raw_lines[example_idx * 4].strip('\n')
            query = raw_lines[example_idx * 4 + 1].strip('\n')
            y = raw_lines[example_idx * 4 + 2].strip('\n')

            if len(y.split()) > 1:
                y = hardcode_y(y)
                # c[len(y.split())]+=1#
                dataset.append((table_column_name + ' ' + query, y, len(table_column_name.split())))
    print("Loaded %i examples from file %s" % (n_examples, filename))

    return dataset


def hardcode_y(y):
    y_toc = y.split()
    y_toc_len = len(y_toc)
    s = y_toc[1]

    if y_toc_len == 3:
        if s == 'select':
            sql_y = y_toc[1] + ' <GO> ' + y_toc[2] + ' from' + ' ' + y_toc[0]
        else:
            sql_y = 'select ' + y_toc[1] + ' ' + y_toc[2] + ' from' + ' ' + y_toc[0]

    elif y_toc_len == 6:
        if s == 'select':
            sql_y = y_toc[1] + ' <GO> ' + y_toc[2] + ' from ' + y_toc[0] + ' where ' + ' '.join(y_toc[-3:])
        else:
            sql_y = 'select ' + y_toc[1] + ' ' + y_toc[2] + ' from ' + y_toc[0] + ' where ' + ' '.join(y_toc[-3:])

    elif y_toc_len == 9:
        if s == 'select':
            sql_y = y_toc[1] + ' <GO> ' + y_toc[2] + ' from ' + y_toc[0] + ' where ' + ' '.join(
                y_toc[-6:-3]) + ' and ' + ' '.join(y_toc[-3:])
        else:
            sql_y = 'select ' + y_toc[1] + ' ' + y_toc[2] + ' from ' + y_toc[0] + ' where ' + ' '.join(
                y_toc[-6:-3]) + ' and ' + ' '.join(y_toc[-3:])

    elif y_toc_len == 12:
        if s == 'select':
            sql_y = y_toc[1] + ' <GO> ' + y_toc[2] + ' from ' + y_toc[0] + ' where ' + ' '.join(
                y_toc[-9:-6]) + ' and ' + ' '.join(y_toc[-6:-3]) + ' and ' + ' '.join(y_toc[-3:])
        else:
            sql_y = 'select ' + y_toc[1] + ' ' + y_toc[2] + ' from ' + y_toc[0] + ' where ' + ' '.join(
                y_toc[-9:-6]) + ' and ' + ' '.join(y_toc[-6:-3]) + ' and ' + ' '.join(y_toc[-3:])

    elif y_toc_len == 15:
        if s == 'select':
            sql_y = y_toc[1] + ' <GO> ' + y_toc[2] + ' from ' + y_toc[0] + ' where ' + ' '.join(
                y_toc[-12:-9]) + ' and ' + ' '.join(y_toc[-9:-6]) + ' and ' + ' '.join(
                y_toc[-6:-3]) + ' and ' + ' '.join(y_toc[-3:])
        else:
            sql_y = 'select ' + y_toc[1] + ' ' + y_toc[2] + ' from ' + y_toc[0] + ' where ' + ' '.join(
                y_toc[-12:-9]) + ' and ' + ' '.join(y_toc[-9:-6]) + ' and ' + ' '.join(
                y_toc[-6:-3]) + ' and ' + ' '.join(y_toc[-3:])

    return sql_y


class WordEmbeddings:
    """
    Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
    word in the indexer. The 0 vector is returned if an unknown word is queried.
    """

    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
        """
        Returns the embedding for a given word
        :param word: The word to look up
        :return: The UNK vector if the word is not in the Indexer or the vector otherwise
        """
        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]


def load_word_vecs(word_vecs_filename: str):
    f = open(word_vecs_filename)
    word_indexer = Indexer()
    vectors = []
    # Make position 0 a PAD token, which can be useful if you
    word_indexer.add_and_get_index("PAD")
    # Make position 1 the UNK token
    word_indexer.add_and_get_index("UNK")
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx + 1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word_indexer.add_and_get_index(word)
            # Append the PAD and UNK vectors to start. Have to do this weirdly because we need to read the first line
            # of the file to see what the embedding dim is
            if len(vectors) == 0:
                vectors.append(np.zeros(vector.shape[0]))
                vectors.append(np.zeros(vector.shape[0]))
            vectors.append(vector)
    f.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    # Turn vectors into a 2-D numpy array
    return WordEmbeddings(word_indexer, np.array(vectors))


def tokenize(x) -> List[str]:
    """
    :param x: string to tokenize
    :return: x tokenized with whitespace tokenization
    """
    return x.split()


def index(x_tok: List[str], indexer: Indexer) -> List[int]:
    return [indexer.index_of(xi) if indexer.index_of(xi) >= 0 else indexer.index_of(UNK_SYMBOL) for xi in x_tok]


def index_data(data, input_indexer: Indexer, output_indexer: Indexer, example_len_limit):
    """
    Indexes the given data
    :param data:
    :param input_indexer:
    :param output_indexer:
    :param example_len_limit:
    :return:
    """
    data_indexed = []
    for (x, y,
         z) in data:  # data[1] = ('1-1000181-1 state/territory text/background^colour format current^slogan current^series notes what is the current^series where the notes new^series^began^in^june^2011 ?', '1-1000181-1 select current^series notes = new^series^began^in^june^2011', 7)
        x_tok = tokenize(x)
        y_tok = tokenize(y)[0:example_len_limit]
        data_indexed.append(Example(x, x_tok, index(x_tok, input_indexer), y, y_tok,
                                    index(y_tok, output_indexer) + [output_indexer.index_of(EOS_SYMBOL)], z))
    return data_indexed


def index_datasets(word_vectors, train_data, dev_data, test_data, example_len_limit, unk_threshold=0.0,
                   use_pretrained=True) -> (List[Example], List[Example], List[Example], Indexer, Indexer):
    """
    Indexes train and test datasets where all words occurring less than or equal to unk_threshold times are
    replaced by UNK tokens.
    :param train_data:
    :param dev_data:
    :param test_data:
    :param example_len_limit:
    :param unk_threshold: threshold below which words are replaced with unks. If 0.0, the model doesn't see any
    UNKs at train time
    :return:
    """
    input_word_counts = Counter()

    if use_pretrained == True:
        input_indexer = word_vectors.word_indexer  # Indexer()
        output_indexer = word_vectors.word_indexer  # Indexer()#

        input_indexer.add_and_get_index(PAD_SYMBOL)
        input_indexer.add_and_get_index(UNK_SYMBOL)
        output_indexer.add_and_get_index(PAD_SYMBOL)
        output_indexer.add_and_get_index(SOS_SYMBOL)
        output_indexer.add_and_get_index(EOS_SYMBOL)

        # # Count words and build the indexers
        # for (x, y, z) in train_data:
        #     for word in tokenize(x):
        #         input_word_counts[word] += 1.0

        # # Index all input words above the UNK threshold
        # for word in input_word_counts.keys():
        #     if input_word_counts[word] > unk_threshold + 0.5:
        #         input_indexer.add_and_get_index(word) 

    else:
        # Count words and build the indexers
        for (x, y, z) in train_data:
            for word in tokenize(x):
                input_word_counts[word] += 1.0

        input_indexer = Indexer()
        output_indexer = Indexer()

        # Reserve 0 for the pad symbol for convenience
        input_indexer.add_and_get_index(PAD_SYMBOL)
        input_indexer.add_and_get_index(UNK_SYMBOL)
        output_indexer.add_and_get_index(PAD_SYMBOL)
        output_indexer.add_and_get_index(SOS_SYMBOL)
        output_indexer.add_and_get_index(EOS_SYMBOL)
        output_indexer.add_and_get_index(UNK_SYMBOL)
        # output_indexer.add_and_get_index("where")
        # output_indexer.add_and_get_index("from")
        # output_indexer.add_and_get_index("and")

        # Index all input words above the UNK threshold
        for word in input_word_counts.keys():
            if input_word_counts[word] > unk_threshold + 0.5:
                input_indexer.add_and_get_index(word)

                # Index all output tokens in train
    for (x, y, z) in train_data:
        for y_tok in tokenize(y):
            output_indexer.add_and_get_index(y_tok)
    # Index things
    train_data_indexed = index_data(train_data, input_indexer, output_indexer, example_len_limit)
    dev_data_indexed = index_data(dev_data, input_indexer, output_indexer, example_len_limit)
    test_data_indexed = index_data(test_data, input_indexer, output_indexer, example_len_limit)
    return train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer


def load_datasets2(train_path: str, dev_path: str, test_path: str) -> (
        List[Tuple[str, str, str]], List[Tuple[str, str, str]], List[Tuple[str, str, str]]):
    # load_datasets2 does not concatenate table column names and query
    train_raw = load_dataset2(train_path)
    dev_raw = load_dataset2(dev_path)
    test_raw = load_dataset2(test_path)
    return train_raw, dev_raw, test_raw


def load_dataset2(filename: str) -> List[Tuple[str, str, str]]:
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


def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int,
                             reverse_input=False) -> np.ndarray:
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
    return np.array(
        [[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)]
         for ex in exs])
