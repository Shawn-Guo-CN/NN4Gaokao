import theano
import theano.tensor as T
import numpy as np

from layers import *

def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

class LSTM_LR_model(object):
    def __init__(self, x, y, mask, emb, word_size=100, hidden_size=400, out_size=2,
                 use_dropout=True, drop_p=0.5, mean_pooling=True, prefix='model_'):
        self.name = 'LSTM_LR'

        self.embedd_layer = Embedding_layer(
            x=x,
            emb=emb,
            word_size=word_size,
            prefix='embedd_layer_'
        )

        self.lstm_layer = LSTM_layer(
            x=self.embedd_layer.output,
            mask=T.transpose(mask),
            in_size=word_size,
            hidden_size=hidden_size,
            mean_pooling=mean_pooling,
            prefix='lstm0_'
        )

        if use_dropout:
            self.dropout_layer = Dropout_layer(x=self.lstm_layer.output, p=drop_p)

            self.lr_layer = LogisticRegression(
                x=self.dropout_layer.output,
                y=y,
                in_size=hidden_size,
                out_size=out_size
            )
        else:
            self.lr_layer = LogisticRegression(
                x=self.lstm_layer.output,
                y=y,
                in_size=hidden_size,
                out_size=out_size
            )

        self.output = self.lr_layer.y_d

        self.p_d = self.lr_layer.y_given_x[:, 1]

        self.error = self.lr_layer.error

        self.loss = self.lr_layer.loss

        self.params = dict(self.embedd_layer.params.items()+
                           self.lstm_layer.params.items()+
                           self.lr_layer.params.items()
                           )