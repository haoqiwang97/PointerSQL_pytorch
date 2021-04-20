from utils import *
from typing import List, Tuple
from collections import Counter


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
    def __init__(self, x: str, x_tok: List[str], x_indexed: List[int], y, y_tok, y_indexed):
        self.x = x
        self.x_tok = x_tok
        self.x_indexed = x_indexed
        self.y = y
        self.y_tok = y_tok
        self.y_indexed = y_indexed

    def __repr__(self):
        return " ".join(self.x_tok) + " => " + " ".join(self.y_tok) + "\n   indexed as: " + repr(self.x_indexed) + " => " + repr(self.y_indexed)

    def __str__(self):
        return self.__repr__()
    
    
PAD_SYMBOL = "<PAD>"
UNK_SYMBOL = "<UNK>"
SOS_SYMBOL = "<SOS>"
EOS_SYMBOL = "<EOS>"

    
def load_datasets(train_path: str, dev_path: str, test_path: str) -> (List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]):
    train_raw = load_dataset(train_path)
    dev_raw = load_dataset(dev_path)
    test_raw = load_dataset(test_path)
    return train_raw, dev_raw, test_raw


def load_dataset(filename: str) -> List[Tuple[str, str]]:
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

            dataset.append((table_column_name + ' ' + query, y))
    print("Loaded %i examples from file %s" % (n_examples, filename))
    return dataset


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
    for (x, y) in data:
        x_tok = tokenize(x)
        y_tok = tokenize(y)[0:example_len_limit]
        data_indexed.append(Example(x, x_tok, index(x_tok, input_indexer), y, y_tok,
                                          index(y_tok, output_indexer) + [output_indexer.index_of(EOS_SYMBOL)]))
    return data_indexed


def index_datasets(train_data, dev_data, test_data, example_len_limit, unk_threshold=0.0) -> (List[Example], List[Example], List[Example], Indexer, Indexer):
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
    # Count words and build the indexers
    for (x, y) in train_data:
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
    # Index all input words above the UNK threshold
    for word in input_word_counts.keys():
        if input_word_counts[word] > unk_threshold + 0.5:
            input_indexer.add_and_get_index(word)
    # Index all output tokens in train
    for (x, y) in train_data:
        for y_tok in tokenize(y):
            output_indexer.add_and_get_index(y_tok)
    # Index things
    train_data_indexed = index_data(train_data, input_indexer, output_indexer, example_len_limit)
    dev_data_indexed = index_data(dev_data, input_indexer, output_indexer, example_len_limit)
    test_data_indexed = index_data(test_data, input_indexer, output_indexer, example_len_limit)
    return train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer


def load_datasets2(train_path: str, dev_path: str, test_path: str) -> (List[Tuple[str, str, str]], List[Tuple[str, str, str]], List[Tuple[str, str, str]]):
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