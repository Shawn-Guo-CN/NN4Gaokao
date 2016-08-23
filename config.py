from collections import OrderedDict
import optimizers
import models

dataset = 'data/GKHMC.pickle'

word2id_param_file = 'data/params/word2id.pickle'
id2word_param_file = 'data/params/id2word.pickle'
keys_param_file = 'data/params/keys.pickle'
id2vec_param_file = 'data/params/id2vec.pickle'
embedding_file = 'data/raw_material/dict.txt'
embedding_param_file = 'data/params/embedding_matrix.pickle'

unk = 51418


model_config = {
    'learning_rate' : 1.0,
    'word_size' : 100,
    'hidden_size' : 1500,
    'max_epoch' : 50,
    'lr_decay' : 1 / 1.15,
    'batch_size' : 256,
    'vocab_size' : 51418
}

options = OrderedDict(
    {
        'model':models.LSTM_LR_model, # define the model
        'word_size':100, # input dimension
        'hidden_size':1000,  # number of hidden units in single layer
        'out_size':2, # number of units in output layer
        'patience':10,  # Number of epoch to wait before early stop if no progress
        'max_epochs':10000,  # The maximum number of epoch to run
        'dispFreq':10,  # Display to stdout the training progress every N updates
        'decay_c':0.1,  # Weight decay for the classifier applied to the U weights.
        'lrate':1,  # Learning rate for sgd (not used for adadelta and rmsprop)
        'optimizer':optimizers.adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        'valid_freq':5,  # Compute the validation error after this number of update.
        'maxlen':100,  # Sequence longer then this get ignored
        'batch_size':1800,  # The batch size during training.
        'valid_batch_size':20,  # The batch size used for validation/test set.
        'dataset':'gkhmc',
        'nkernals':[20, 50],
    }
)
