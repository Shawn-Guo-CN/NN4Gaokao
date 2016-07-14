import cPickle

import numpy as np
import tensorflow as tf
import uniout

def build_dict_vocab(embedding_file):
    print "... running reader.build_word_dict"

    with open(embedding_file, mode="r") as datafile:
        id_to_vec = {}
        word_to_id = {}
        id_to_word = {}
        embedding_matrix = []
        keys = []
        i = 0
        for line in datafile:
            line = line[:-1]
            line = line.strip()
            words = line.split(" ")
            vec = [float(word) for word in words[1:]]
            if words[0] in keys:
                continue
            keys.append(words[0])
            word_to_id[words[0]] = i
            id_to_word[i] = words[0]
            id_to_vec[i] = np.asarray(vec, dtype=float)
            embedding_matrix.append(id_to_vec[i])
            i += 1
        id_to_vec[i] = np.asarray([1.] * 100, dtype=float)
        id_to_word[i] = 'unk'
        embedding_matrix.append(id_to_vec[i])
        embedding_matrix = np.asarray(embedding_matrix, dtype=float)

    return word_to_id, id_to_word, keys, id_to_vec, embedding_matrix

def data_to_word_ids(file_name, word_to_id, keys):
    print '... loading data to ids'

    unk = len(word_to_id)
    data_list = []
    with open(file_name, 'r') as f:
        for line in f:
            vec = []
            for word in line.strip().split(" "):
                if word in keys:
                    vec.append(int(word_to_id[word]))
                else:
                    vec.append(unk)
            data_list.append(vec)
    data_list = np.asarray(data_list)
    return data_list

def lable_to_var(file_name):
    print '... load file to labels'

    y = []
    with open(file_name, 'r') as f:
        for line in f:
            y.append(int(line.strip()))
    y = np.asarray(y)
    return y

def init_and_save():
    embedding_file = 'data/dict.txt'
    x_file = 'data/train_words.txt'
    y_file = 'data/train_labels.txt'

    word_to_id, id_to_word, keys, id_to_vec, embedding_matrix = build_dict_vocab(embedding_file)
    with open('data/params/word2id.dat', 'wb') as f:
        print '... saving word_to_id to data/params/word2id.dat'
        cPickle.dump(word_to_id, f)
    with open('data/params/id2word.dat', 'wb') as f:
        print '... saving word_to_id to data/params/id2word.dat'
        cPickle.dump(id_to_word, f)
    with open('data/params/keys.dat', 'wb') as f:
        print '... saving keys to data/params/keys.dat'
        cPickle.dump(keys, f)
    with open('data/params/id2vec.dat', 'wb') as f:
        print '... saving id_to_vec to data/params/id2vec.dat'
        cPickle.dump(id_to_vec, f)
    with open('data/params/embedding_matrix', 'wb') as f:
        print '... saving embedding_matrix to data/params/embedding_matrix.dat'
        cPickle.dump(embedding_matrix, f)

    x = data_to_word_ids(x_file, word_to_id, keys)
    with open('data/params/x.dat', 'wb') as f:
        print '... saving x to data/params/x.dat'
        cPickle.dump(x, f)

    y = lable_to_var(y_file)
    with open('data/params/y.dat', 'wb') as f:
        print '... saving y to ' + 'data/params/y.dat'
        cPickle.dump(y, f)

def get_embedding_matrix_from_param_file(file_name):
    with open(file_name, 'rb') as f:
        print '... saving embedding_matrix to data/params/embedding_matrix.dat'
        embedding_matrix = cPickle.load(f)

    return embedding_matrix

def load_params():
    """
    with open('data/params/word2id.dat', 'rb') as f:
        print '... loading word_to_id'
        word_to_id = cPickle.load(f)
    with open('data/params/id2word.dat', 'rb') as f:
        print '... loading id_to_word'
        id_to_word = cPickle.load(f)
    with open('data/params/keys.dat', 'rb') as f:
        print '... loading keys '
        keys = cPickle.load(f)
    with open('data/params/id2vec.dat', 'rb') as f:
        print '... loading id_to_vec'
        id_to_vec = cPickle.load(f)
    with open('data/params/embedding_matrix', 'rb') as f:
        print '... loading embedding_matrix'
        embedding_matrix = cPickle.load(f)

    for i in xrange(len(x)):
        x[i].append(y[i])

    x = sorted(x, key=len)
    x = np.asarray(x)

    for i in xrange(len(x)):
        y[i] = x[i][-1]
        x[i] = x[i][:-1]

    with open('data/params/x.dat', 'wb') as f:
        print '... saving x to data/params/x.dat'
        cPickle.dump(x, f)

    with open('data/params/y.dat', 'wb') as f:
        print '... saving y to ' + 'data/params/y.dat'
        cPickle.dump(y, f)
    """
    with open('data/params/x.dat', 'rb') as f:
        print '... loading x'
        x = cPickle.load(f)
    with open('data/params/y.dat', 'rb') as f:
        print '... loading y'
        y = cPickle.load(f)

    return x, y

def xy_iterator(batch_size):
    x, y = load_params()
    unk = 51418
    data_num = len(y)
    batch_num = data_num // batch_size
    if batch_num == 0:
        raise ValueError("batch_num == 0, decrease batch_size")


    for i in range(batch_num):
        batch_end = batch_size * (i + 1) - 1
        batch_len = len(x[batch_end])
        data = np.zeros([batch_size, batch_len], dtype=np.int32)
        for ii in range(batch_size):
            data[ii] = x[batch_size * i + ii] + [unk] * (batch_len - len(x[batch_size * i + ii]))
        yield data, y[batch_size * i : batch_size * (i + 1)]

if __name__ == "__main__":
    # init_and_save()
    # load_params()
    for i in xy_iterator(12):
        print "x:", i[0]
        print "y:", i[1]