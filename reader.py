import theano
import numpy as np

import cPickle
import config

def build_dict_vocab(embedding_file):
    print "... running reader.build_word_dict"

    with open(embedding_file, mode="r") as datafile:
        id_to_vec = {}
        word_to_id = {}
        id_to_word = {}
        embedding_matrix = []
        keys = []
        embedding_matrix.append(np.asarray([0.] * 100, dtype=float))
        i = 1
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
        id_to_word[i] = '<unk>'
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

def data_to_word_ids_qla(file_name, word_to_id, keys, step=8):
    print '... loading data to ids'

    unk = len(word_to_id)
    q_list = []
    l_list = []
    a_list = []
    with open(file_name, 'r') as f:
        line_num = 0
        current_q = None
        current_l = None
        for line in f:
            line_num += 1
            if line_num % step == 1:
                vec = []
                for word in line.strip().split(" "):
                    if word in keys:
                        vec.append(int(word_to_id[word]))
                    else:
                        vec.append(unk)
                current_q = vec
            elif line_num % step == 2:
                vec = []
                for word in line.strip().split(" "):
                    if word in keys:
                        vec.append(int(word_to_id[word]))
                    else:
                        vec.append(unk)
                current_l = vec
            else:
                vec = []
                for word in line.strip().split(" "):
                    if word in keys:
                        vec.append(int(word_to_id[word]))
                    else:
                        vec.append(unk)
                q_list.append(current_q)
                l_list.append(current_l)
                a_list.append(vec)

    return q_list, l_list, a_list

def lable_to_var(file_name):
    print '... load file to labels'

    y = []
    with open(file_name, 'r') as f:
        for line in f:
            y.append(int(line.strip()))
    y = np.asarray(y)
    return y

def init_and_save():

    word_to_id, id_to_word, keys, id_to_vec, embedding_matrix = build_dict_vocab(config.embedding_file)
    with open(config.word2id_param_file, 'wb') as f:
        print '... saving word_to_id to data/params/word2id.dat'
        cPickle.dump(word_to_id, f)
    with open(config.id2word_param_file, 'wb') as f:
        print '... saving word_to_id to data/params/id2word.dat'
        cPickle.dump(id_to_word, f)
    with open(config.keys_param_file, 'wb') as f:
        print '... saving keys to data/params/keys.dat'
        cPickle.dump(keys, f)
    with open(config.id2vec_param_file, 'wb') as f:
        print '... saving id_to_vec to data/params/id2vec.dat'
        cPickle.dump(id_to_vec, f)
    with open(config.embedding_param_file, 'wb') as f:
        print '... saving embedding_matrix to data/params/embedding_matrix.dat'
        cPickle.dump(embedding_matrix, f)

    train_q, train_l, train_a = data_to_word_ids_qla('data/tiku_qla/train_words.txt', word_to_id, keys, 8)
    train_y = lable_to_var('data/tiku_qla/train_labels.txt')
    train_set = [train_q, train_l, train_a, train_y]

    test_q, test_l, test_a = data_to_word_ids_qla('data/tiku_qla/test_words.txt', word_to_id, keys, 6)
    test_y = lable_to_var('data/tiku_qla/test_labels.txt')
    test_set = [test_q, test_l, test_a, test_y]

    with open(config.dataset, 'wb') as f:
        print '... saving dataset to data/GKHMC_qla.pickle'
        cPickle.dump([train_set, test_set], f)


def get_embedding_matrix_from_param_file(file_name):
    with open(file_name, 'rb') as f:
        print '... loadng embedding_matrix from ' + file_name
        embedding_matrix = cPickle.load(f)

    return embedding_matrix

def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.

    labels = np.asarray(labels, dtype='int32')

    return x, x_mask, labels

def prepare_data_qla(questions, leads, answers, labels):

    q_lengths = [len(s) for s in questions]
    l_lengths = [len(s) for s in leads]
    a_lengths = [len(s) for s in answers]

    n_samples = len(labels)
    q_maxlen = np.max(q_lengths)
    l_maxlen = np.max(l_lengths)
    a_maxlen = np.max(a_lengths)

    q = np.zeros((n_samples, q_maxlen)).astype('int32')
    q_mask = np.zeros((n_samples, q_maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(questions):
        q[idx, :q_lengths[idx]] = s
        q_mask[idx, :q_lengths[idx]] = 1.

    l = np.zeros((n_samples, l_maxlen)).astype('int32')
    l_mask = np.zeros((n_samples, l_maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(leads):
        l[idx, :l_lengths[idx]] = s
        l_mask[idx, :l_lengths[idx]] = 1.

    a = np.zeros((n_samples, a_maxlen)).astype('int32')
    a_mask = np.zeros((n_samples, a_maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(answers):
        a[idx, :a_lengths[idx]] = s
        a_mask[idx, :a_lengths[idx]] = 1.

    labels = np.asarray(labels, dtype='int32')

    return q, q_mask, l, l_mask, a, a_mask, labels

initialize = True
datasets = []
train_set_x = []
train_set_y = []
test_set_x = []
test_set_y = []

def gkhmc_iterator(path="data/GKHMC.pickle", is_train=True, batch_size=100):
    global initialize
    global datasets
    global train_set_x
    global train_set_y
    global test_set_x
    global test_set_y

    if initialize:
        with open(path, 'rb') as f:
            datasets = cPickle.load(f)
        train_set_x, train_set_y = datasets[0]
        test_set_x, test_set_y = datasets[1]
        initialize = False

    train_batch_num = len(train_set_x) / batch_size
    test_batch_num = len(test_set_x) / batch_size

    if is_train:
        for idx in xrange(train_batch_num):
            yield prepare_data(train_set_x[idx * batch_size : (idx + 1) * batch_size],
                               train_set_y[idx * batch_size : (idx + 1) * batch_size])
    else:
        for idx in xrange(test_batch_num):
            yield prepare_data(test_set_x[idx * batch_size : (idx + 1) * batch_size],
                              test_set_y[idx * batch_size : (idx + 1) * batch_size])

initialize_qla = True
train_set_q = []
train_set_l = []
train_set_a = []
test_set_q = []
test_set_l = []
test_set_a = []

def gkhmc_qla_iterator(path="data/GKHMC_qla.pickle", is_train=True, batch_size=100):
    global initialize_qla
    global datasets
    global train_set_q
    global train_set_l
    global train_set_a
    global train_set_y
    global test_set_q
    global test_set_l
    global test_set_a
    global test_set_y

    if initialize_qla:
        with open(path, 'rb') as f:
            datasets = cPickle.load(f)
        train_set_q, train_set_l, train_set_a, train_set_y = datasets[0]
        test_set_q, test_set_l, test_set_a, test_set_y = datasets[1]
        initialize_qla = False

    train_batch_num = len(train_set_a) / batch_size
    test_batch_num = len(test_set_a) / batch_size

    if is_train:
        for idx in xrange(train_batch_num):
            yield prepare_data_qla(train_set_q[idx * batch_size: (idx + 1) * batch_size],
                                   train_set_l[idx * batch_size: (idx + 1) * batch_size],
                                   train_set_a[idx * batch_size: (idx + 1) * batch_size],
                                   train_set_y[idx * batch_size: (idx + 1) * batch_size])
    else:
        for idx in xrange(test_batch_num):
            yield prepare_data_qla(test_set_q[idx * batch_size: (idx + 1) * batch_size],
                                   test_set_l[idx * batch_size: (idx + 1) * batch_size],
                                   test_set_a[idx * batch_size: (idx + 1) * batch_size],
                                   test_set_y[idx * batch_size: (idx + 1) * batch_size])

if __name__ == "__main__":
    # init_and_save()

    embedd = get_embedding_matrix_from_param_file('data/qla_params/embedding_matrix.pickle')
    print embedd[0]

    for i in gkhmc_qla_iterator(is_train=False, batch_size=20):
        print i[6]