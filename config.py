from collections import OrderedDict
import optimizers
import models

dataset = 'data/GKHMC_qla.pickle'

word2id_param_file = 'data/qla_params/word2id.pickle'
id2word_param_file = 'data/qla_params/id2word.pickle'
keys_param_file = 'data/qla_params/keys.pickle'
id2vec_param_file = 'data/qla_params/id2vec.pickle'
embedding_file = 'data/tiku_qla/dict.txt'
embedding_param_file = 'data/qla_params/embedding_matrix.pickle'

unk = 51418

options = OrderedDict(
    {
        'model': models.DMN,  # define the model
        'word_size': 100,  # input dimension
        'hidden_size': 400,  # number of hidden units in single layer
        'out_size': 2,  # number of units in output layer
        'patience': 10,  # Number of epoch to wait before early stop if no progress
        'max_epochs': 10000,  # The maximum number of epoch to run
        'lrate': 0.1,  # Learning rate for sgd (not used for adadelta and rmsprop)
        'optimizer': optimizers.adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        'valid_freq': 5,  # Compute the validation error after this number of update.
        'maxlen': 100,  # Sequence longer then this get ignored
        'batch_size': 1200,  # The batch size during training.
        'valid_batch_size': 40,  # The batch size used for validation/test set.
        'dataset': 'gkhmc_qla',
        'param_path': 'data/',  # path to save parameters
        'loaded_params': 'data/DMN_mem800_lrate1_batch1200_epoch8_perform50.83.pickle',
        'use_dropout': True,  # use dropout layer or not
        'drop_p': 0.5,  # the probability of dropout
        'lstm_mean_pooling': False,  # use mean pooling as output of lstm or not
        'mem_size': 800  # the hidden size of memory
    }
)
