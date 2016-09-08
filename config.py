from collections import OrderedDict
import optimizers
import models

dataset = 'data/GKHMC_qla.pickle'

word2id_param_file = 'data/qla_params/word2id.pickle'
id2word_param_file = 'data/qla_params/id2word.pickle'
keys_param_file = 'data/qla_params/keys.pickle'
id2vec_param_file = 'data/qla_params/id2vec.pickle'
kb_param_file = 'data/qla_params/kb.pickle'
embedding_file = 'data/raw_material/dict.txt'
textbook_file = 'data/raw_material/textbook_mainwords.txt'
embedding_param_file = 'data/qla_params/embedding_matrix.pickle'

train_words_file = 'data/raw_material/train_words.txt'
train_labels_file = 'data/raw_material/train_labels.txt'
test_words_file = 'data/raw_material/test_words.txt'
test_labels_file = 'data/raw_material/test_labels.txt'

unk = 51418

options = OrderedDict(
    {
        'model': models.Memory_Network,  # define the model
        'word_size': 100,  # input dimension
        'hidden_size': 400,  # number of hidden units in single layer
        'out_size': 2,  # number of units in output layer
        'patience': 10,  # Number of epoch to wait before early stop if no progress
        'max_epochs': 100000,  # The maximum number of epoch to run
        'lrate': 0.1,  # Learning rate for sgd (not used for adadelta and rmsprop)
        'optimizer': optimizers.rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        'valid_freq': 20,  # Compute the validation error after this number of update.
        'maxlen': 100,  # Sequence longer then this get ignored
        'batch_size': 1000,  # The batch size during training.
        'valid_batch_size': 80,  # The batch size used for validation/test set.
        'dataset': 'gkhmc_qla',
        'param_path': 'data/',  # path to save parameters
        'loaded_params': None,
        'use_dropout': True,  # use dropout layer or not
        'drop_p': 0.5,  # the probability of dropout
        'lstm_mean_pooling': False,  # use mean pooling as output of lstm or not
        'mem_size': 400  # the hidden size of memory
    }
)
