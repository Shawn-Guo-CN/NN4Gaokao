import collections
import os
import cPickle

import numpy as np
import tensorflow as tf
import uniout

def build_dict_vocab(embedding_file):
    print "... running reader.build_word_dict"

    with open(embedding_file, mode="r") as datafile:
        id_to_vec = {}
        word_to_id = {}
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
            id_to_vec[i] = np.asarray(vec, dtype=float)
            embedding_matrix.append(id_to_vec[i])
            i += 1
        id_to_vec[i] = np.asarray([1.] * 100, dtype=float)
        embedding_matrix.append(id_to_vec[i])
        embedding_matrix = np.asarray(embedding_matrix, dtype=float)

    return word_to_id, keys, id_to_vec, embedding_matrix

def data_to_word_ids(file_name, word_to_id, keys):
    print '... reading data to ids'

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

def test():
    embedding_file = 'data/dict.txt'
    x_file = 'data/train_words.txt'
    word_to_id, keys, id_to_vec, embedding_matrix = build_dict_vocab(embedding_file)
    with open('data/params/word2id.dat', 'wb') as f:
        print '... saving word_to_id to data/params/word2id.dat'
        cPickle.dump(word_to_id, f)
    with open('data/params/keys.dat', 'wb') as f:
        print '... saving keys to data/params/keys.dat'
        cPickle.dump(keys, f)
    with open('data/params/id2vec.dat', 'wb') as f:
        print '... saving id_to_vec to data/params/id2vec.dat'
        cPickle.dump(id_to_vec, f)
    with open('data/params/embedding_matrix', 'wb') as f:
        print '... saving embedding_matrix to data/params/embedding_matrix.dat'
        cPickle.dump(embedding_matrix, f)
    print keys[2]
    print id_to_vec[2]
    print embedding_matrix[2]
    x = data_to_word_ids(x_file, word_to_id, keys)
    with open('data/params/x.dat', 'wb') as f:
        print '... saving x to data/params/x.dat'
        cPickle.dump(x, f)
    print x.shape
    print x[0]

if __name__ == "__main__":
    test()