import os
import re
import collections

import numpy as np
import tensorflow as tf

DATA_FOLDER = 'data'
DATA_FILE = 'input.txt'

FILENAME = os.path.join(DATA_FOLDER, DATA_FILE)
vocab_size = 50000


def _read_words():
    with tf.gfile.GFile(FILENAME, 'r') as f:
        return f.read().decode('utf-8').replace('\n', ' <EOS> ').split()


def _build_dataset():
    words = _read_words()

    counter = collections.Counter(words)

    global vocab_size
    if len(counter) < vocab_size:
        vocab_size = len(counter)

    count = [['<UNK>', -1]]
    count.extend(counter.most_common(vocab_size - 1))

    dictionary = dict()
    for i, (word, _) in enumerate(count):
        dictionary[word] = i

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reverse_dictionary

data, count, dict, reverse_dict = _build_dataset()


def gen_batches(data, batch_size, seq_length):
    """Generates batches with the shape (batch_size, seq_length, 1).
    'data' is a list of integers, where each integer is an id for a word.

    Also rearranges data so batches process sequential data.

    If we have the dataset:
    x_in = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

    and batch_size is 2 and seq_len is 3. Then the dataset is
    reordered such that:

                   Batch 1    Batch 2
                 ------------------------
    batch pos 1  [1, 2, 3]   [4, 5, 6]
    batch pos 2  [7, 8, 9]   [10, 11, 12]

    This ensures that we use the last hidden state of batch 1 to initialize
    batch 2."""
    # if data doesn't fit evenly into the batches, then cut off the end
    words_in_batch = batch_size * seq_length
    end_idx = words_in_batch * (len(data) / words_in_batch)

    x = np.array(data[:end_idx]).reshape(batch_size, -1)
    y_list = data[1: end_idx]; y_list.append(data[0])  # last element is wrong!
    y = np.array(y_list).reshape((batch_size, -1))

    x_batches = np.split(x, x.shape[1] / seq_length, 1)
    y_batches = np.split(y, y.shape[1] / seq_length, 1)

    for x, y in zip(x_batches, y_batches):
        yield x, y


def split_data(tf):
    """Splits data into train and val set. 'tf' is the fraction of train data."""
    assert 0 < tf <= 1

    train_n = int(len(data) * tf)
    tr_data = data[:train_n]
    val_data = data[train_n:]

    return tr_data, val_data
