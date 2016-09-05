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

class GRU_LR_model(object):
    def __init__(self, x, y, mask, emb, word_size=100, hidden_size=400, out_size=2,
                 use_dropout=True, drop_p=0.5, mean_pooling=True, prefix='model_'):
        self.name = 'GRU_LR'

        self.embedd_layer = Embedding_layer(
            x=x,
            emb=emb,
            word_size=word_size,
            prefix='embedd_layer_'
        )

        self.lstm_layer = GRU_layer(
            x=self.embedd_layer.output,
            mask=T.transpose(mask),
            in_size=word_size,
            hidden_size=hidden_size,
            mean_pooling=mean_pooling,
            prefix='gru0_'
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

class Memory_Network(object):
    def __init__(self, q, l, a, y, emb, mem_size=200, word_size=100, prefix='mem_nn_'):
        self.name = 'MEM_NN'

        # L2-normalize the embedding matrix
        emb_ = np.sqrt(np.sum(emb ** 2, axis=1))
        emb = emb / np.dot(emb_.reshape(-1, 1), np.ones((1, emb.shape[1])))
        emb[0, :] = 0.

        self.emb = theano.shared(
            value=np.asarray(emb, dtype=theano.config.floatX),
            name=prefix + 'emb',
            borrow=True
        )

        self.A = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(6. / (mem_size + word_size)),
                high=np.sqrt(6. / (mem_size + word_size)),
                size=(word_size, mem_size)
            ).astype(theano.config.floatX),
            name=prefix+'A',
            borrow=True
        )

        self.B = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(6. / (mem_size + word_size)),
                high=np.sqrt(6. / (mem_size + word_size)),
                size=(word_size, mem_size)
            ).astype(theano.config.floatX),
            name=prefix+'B',
            borrow=True
        )

        self.C = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(6. / (mem_size + word_size)),
                high=np.sqrt(6. / (mem_size + word_size)),
                size=(word_size, mem_size)
            ).astype(theano.config.floatX),
            name=prefix + 'C',
            borrow=True
        )

        self.W = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(6. / (mem_size + word_size)),
                high=np.sqrt(6. / (mem_size + word_size)),
                size=(mem_size, word_size)
            ).astype(theano.config.floatX),
            name=prefix + 'W',
            borrow=True
        )

        self.q_embedd_layer = Embedding_layer_uniEmb(
            x=T.transpose(q),
            emb=self.emb,
            word_size=word_size,
            prefix=prefix+'q_embedd_layer_'
        )

        self.l_embedd_layer = Embedding_layer_uniEmb(
            x=T.transpose(l),
            emb=self.emb,
            word_size=word_size,
            prefix=prefix+'l_embedd_layer_'
        )

        self.a_embedd_layer = Embedding_layer_uniEmb(
            x=T.transpose(a),
            emb=self.emb,
            word_size=word_size,
            prefix=prefix+'a_embedd_layer_'
        )

        self.r_q = self.q_embedd_layer.output
        self.r_l = T.mean(self.l_embedd_layer.output, axis=1)
        self.r_a = T.mean(self.a_embedd_layer.output, axis=1)

        self.m = T.dot(self.r_q, self.A)
        self.u = T.dot(self.r_l, self.B)

        p = T.nnet.softmax(T.batched_dot(self.m, self.u))
        self.p = T.reshape(p, (p.shape[0], p.shape[1], 1), ndim=3)
        self.c = T.dot(self.r_q, self.C)

        self.o = T.mean(self.c * self.p, axis=1)

        self.lr_layer = LogisticRegression(
            x=T.concatenate([T.dot(self.o, self.W), self.r_a], axis=1),
            y=y,
            in_size=word_size * 2,
            out_size=2,
            prefix=prefix+'lr_layer_'
        )

        self.param = {
            prefix+'A': self.A,
            prefix+'B': self.B,
            prefix+'C': self.C,
            prefix+'W': self.W,
            prefix+'emb': self.emb,
        }

        self.output = self.lr_layer.y_d

        self.p_d = self.lr_layer.y_given_x[:, 1]

        self.error = self.lr_layer.error

        self.loss = self.lr_layer.loss

        self.params = dict(self.param.items() +
                           self.lr_layer.params.items())

    def emb_set_value_zero(self):
        self.emb = T.set_subtensor(self.emb[0:], 0.)
