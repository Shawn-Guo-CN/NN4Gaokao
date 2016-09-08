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

        self.gru_layer = GRU_layer(
            x=self.embedd_layer.output,
            mask=T.transpose(mask),
            in_size=word_size,
            hidden_size=hidden_size,
            mean_pooling=mean_pooling,
            prefix='gru0_'
        )

        if use_dropout:
            self.dropout_layer = Dropout_layer(x=self.gru_layer.output, p=drop_p)

            self.lr_layer = LogisticRegression(
                x=self.dropout_layer.output,
                y=y,
                in_size=hidden_size,
                out_size=out_size
            )
        else:
            self.lr_layer = LogisticRegression(
                x=self.gru_layer.output,
                y=y,
                in_size=hidden_size,
                out_size=out_size
            )

        self.output = self.lr_layer.y_d

        self.p_d = self.lr_layer.y_given_x[:, 1]

        self.error = self.lr_layer.error

        self.loss = self.lr_layer.loss

        self.params = dict(self.embedd_layer.params.items()+
                           self.gru_layer.params.items()+
                           self.lr_layer.params.items()
                           )

class Memory_Network(object):
    def __init__(self, q, l, a, y, emb, mem_size=200, word_size=100, use_dropout=True, drop_p=0.5, prefix='mem_nn_'):
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

        if use_dropout:
            self.dropout_layer = Dropout_layer(x=T.concatenate([T.dot(self.o, self.W), self.r_a], axis=1), p=drop_p)

            self.lr_layer = LogisticRegression(
                x=self.dropout_layer.output,
                y=y,
                in_size=word_size * 2,
                out_size=2,
                prefix=prefix+'lr_layer_'
            )
        else:
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


class DMN(object):
    def __init__(self, q, q_mask, l, l_mask, a, a_mask, y, emb, word_size=100, hidden_size=400,
                 use_dropout=True, drop_p=0.5, prefix='DMN_'):
        self.name = 'DMN'

        # L2-normalize the embedding matrix
        emb_ = np.sqrt(np.sum(emb ** 2, axis=1))
        emb = emb / np.dot(emb_.reshape(-1, 1), np.ones((1, emb.shape[1])))
        emb[0, :] = 0.

        self.emb = theano.shared(
            value=np.asarray(emb, dtype=theano.config.floatX),
            name=prefix + 'emb',
            borrow=True
        )

        self.q_embedd_layer = Embedding_layer_uniEmb(
            x=q,
            emb=self.emb,
            word_size=word_size,
            prefix=prefix + 'q_embedd_layer_'
        )

        self.l_embedd_layer = Embedding_layer_uniEmb(
            x=l,
            emb=self.emb,
            word_size=word_size,
            prefix=prefix + 'l_embedd_layer_'
        )

        self.a_embedd_layer = Embedding_layer_uniEmb(
            x=a,
            emb=self.emb,
            word_size=word_size,
            prefix=prefix + 'a_embedd_layer_'
        )

        def _random_weights(x_dim, y_dim):
            return np.random.uniform(
                low=-np.sqrt(6. / (x_dim + y_dim)),
                high=np.sqrt(6. / (x_dim + y_dim)),
                size=(x_dim, y_dim)
            ).astype(theano.config.floatX)

        self.gru_W = theano.shared(
            value=np.concatenate(
                [_random_weights(word_size, hidden_size),
                 _random_weights(word_size, hidden_size),
                 _random_weights(word_size, hidden_size)],
                axis=1
            ).astype(theano.config.floatX),
            name=prefix+'gru_W',
            borrow=True
        )

        self.gru_U = theano.shared(
            value=np.concatenate(
                [_random_weights(hidden_size, hidden_size),
                 _random_weights(hidden_size, hidden_size),
                 _random_weights(hidden_size, hidden_size)],
                axis=1
            ).astype(theano.config.floatX),
            name=prefix+'gru_U',
            borrow=True
        )

        self.gru_B = theano.shared(
            value=np.zeros((3 * hidden_size,)).astype(theano.config.floatX),
            name=prefix+'b',
            borrow=True
        )

        self.q_gru_layer = GRU_layer_uniParam(
            x=self.q_embedd_layer.output,
            W=self.gru_W,
            U=self.gru_U,
            b=self.gru_B,
            mask=T.transpose(q_mask),
            in_size=word_size,
            hidden_size=hidden_size,
            prefix=prefix + 'q_gru_'
        )

        self.l_gru_layer = GRU_layer_uniParam(
            x=self.l_embedd_layer.output,
            W=self.gru_W,
            U=self.gru_U,
            b=self.gru_B,
            mask=T.transpose(l_mask),
            in_size=word_size,
            hidden_size=hidden_size,
            prefix=prefix + 'l_gru_'
        )

        self.a_gru_layer = GRU_layer_uniParam(
            x=self.a_embedd_layer.output,
            W=self.gru_W,
            U=self.gru_U,
            b=self.gru_B,
            mask=T.transpose(a_mask),
            in_size=word_size,
            hidden_size=hidden_size,
            prefix=prefix + 'a_gru_'
        )

        self.e_generate_layer = DMN_GRU(
            x=self.q_gru_layer.out_all,
            l=self.l_gru_layer.output,
            mask=T.transpose(q_mask),
            hidden_size=hidden_size,
            prefix=prefix+'dmn_gru_'
        )

        self.e = self.e_generate_layer.output
        _e = T.dot(self.e, self.gru_U)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        _preact = T.dot(self.l_gru_layer.output, self.gru_U)
        _preact += _e

        _z = T.nnet.sigmoid(_slice(_preact, 0, hidden_size))
        _r = T.nnet.sigmoid(_slice(_preact, 1, hidden_size))
        _c = T.tanh(_slice(_preact, 2, hidden_size) * _r + (T.ones_like(_r) - _r) * _slice(_e, 2, hidden_size))

        self.m = (T.ones_like(_z) - _z) * _c + _z * self.l_gru_layer.output

        if use_dropout:
            self.dropout_layer = Dropout_layer(
                x=T.concatenate([self.m, self.a_gru_layer.output], axis=1),
                p=drop_p)

            self.lr_layer = LogisticRegression(
                x=self.dropout_layer.output,
                y=y,
                in_size=hidden_size * 2,
                out_size=2,
                prefix=prefix+'lr_layer_'
            )
        else:
            self.lr_layer = LogisticRegression(
                x=T.concatenate([self.m, self.a_gru_layer.output], axis=1),
                y=y,
                in_size=hidden_size * 2,
                out_size=2,
                prefix=prefix+'lr_layer_'
            )

        self.param = {
            prefix+'emb': self.emb,
            prefix+'gru_W': self.gru_W,
            prefix+'gru_U': self.gru_U,
            prefix+'gru_b': self.gru_B
        }

        self.output = self.lr_layer.y_d

        self.p_d = self.lr_layer.y_given_x[:, 1]

        self.error = self.lr_layer.error

        self.loss = self.lr_layer.loss

        self.params = dict(
            self.param.items() +
            self.e_generate_layer.params.items() +
            self.lr_layer.params.items()
        )

    def emb_set_value_zero(self):
        self.emb = T.set_subtensor(self.emb[0:], 0.)


class KBMN(object):
    def __init__(self, q, q_mask, l, l_mask, a, a_mask, y, kbm, kbm_mask, emb, word_size=100, hidden_size=400,
                 use_dropout=True, drop_p=0.5, prefix='KBMN_'):
        self.name = 'KBMN'

        self._init_params_(kbm, kbm_mask, emb, word_size=word_size, hidden_size=hidden_size, prefix='KBMN_')

        self.kbm_embedd_layer = Embedding_layer_uniEmb(
            x=self.kbm,
            emb=self.emb,
            word_size=word_size,
            prefix=prefix + 'kbm_embedd_layer_'
        )

        self.q_embedd_layer = Embedding_layer_uniEmb(
            x=q,
            emb=self.emb,
            word_size=word_size,
            prefix=prefix + 'q_embedd_layer_'
        )

        self.l_embedd_layer = Embedding_layer_uniEmb(
            x=l,
            emb=self.emb,
            word_size=word_size,
            prefix=prefix + 'l_embedd_layer_'
        )

        self.a_embedd_layer = Embedding_layer_uniEmb(
            x=a,
            emb=self.emb,
            word_size=word_size,
            prefix=prefix + 'a_embedd_layer_'
        )

        self.kbm_gru_layer = GRU_layer_uniParam(
            x=self.kbm_embedd_layer.output,
            W=self.gru_W,
            U=self.gru_U,
            b=self.gru_B,
            mask=T.transpose(self.kbm_mask),
            in_size=word_size,
            hidden_size=hidden_size,
            prefix=prefix+'kbm_gru_'
        )  # (198,hidden_size)

        self.q_gru_layer = GRU_layer_uniParam(
            x=self.q_embedd_layer.output,
            W=self.gru_W,
            U=self.gru_U,
            b=self.gru_B,
            mask=T.transpose(q_mask),
            in_size=word_size,
            hidden_size=hidden_size,
            prefix=prefix + 'q_gru_'
        )

        self.l_gru_layer = GRU_layer_uniParam(
            x=self.l_embedd_layer.output,
            W=self.gru_W,
            U=self.gru_U,
            b=self.gru_B,
            mask=T.transpose(l_mask),
            in_size=word_size,
            hidden_size=hidden_size,
            prefix=prefix + 'l_gru_'
        )

        self.a_gru_layer = GRU_layer_uniParam(
            x=self.a_embedd_layer.output,
            W=self.gru_W,
            U=self.gru_U,
            b=self.gru_B,
            mask=T.transpose(a_mask),
            in_size=word_size,
            hidden_size=hidden_size,
            prefix=prefix + 'a_gru_'
        )

        self.e_generate_layer = KBMN_GRU(
            x=self.q_gru_layer.out_all,
            l=self.l_gru_layer.output,
            kbm=self.kbm_gru_layer.output,
            mask=T.transpose(q_mask),
            hidden_size=hidden_size,
            prefix=prefix+'kbmn_gru_'
        )

        self.e = self.e_generate_layer.output
        _e = T.dot(self.e, self.gru_U)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        _preact = T.dot(self.l_gru_layer.output, self.gru_U)
        _preact += _e

        _z = T.nnet.sigmoid(_slice(_preact, 0, hidden_size))
        _r = T.nnet.sigmoid(_slice(_preact, 1, hidden_size))
        _c = T.tanh(_slice(_preact, 2, hidden_size) * _r + (T.ones_like(_r) - _r) * _slice(_e, 2, hidden_size))

        self.m = (T.ones_like(_z) - _z) * _c + _z * self.l_gru_layer.output

        if use_dropout:
            self.dropout_layer = Dropout_layer(
                x=T.concatenate([self.m, self.a_gru_layer.output], axis=1),
                p=drop_p)

            self.lr_layer = LogisticRegression(
                x=self.dropout_layer.output,
                y=y,
                in_size=hidden_size * 2,
                out_size=2,
                prefix=prefix+'lr_layer_'
            )
        else:
            self.lr_layer = LogisticRegression(
                x=T.concatenate([self.m, self.a_gru_layer.output], axis=1),
                y=y,
                in_size=hidden_size * 2,
                out_size=2,
                prefix=prefix+'lr_layer_'
            )

        self.param = {
            prefix+'emb': self.emb,
            prefix+'gru_W': self.gru_W,
            prefix+'gru_U': self.gru_U,
            prefix+'gru_b': self.gru_B
        }

        self.output = self.lr_layer.y_d

        self.p_d = self.lr_layer.y_given_x[:, 1]

        self.error = self.lr_layer.error

        self.loss = self.lr_layer.loss

        self.params = dict(
            self.param.items() +
            self.e_generate_layer.params.items() +
            self.lr_layer.params.items()
        )

    def _init_params_(self, kbm, kbm_mask, emb, word_size=100, hidden_size=400, prefix='KBMN_'):
        # L2-normalize the embedding matrix
        emb_ = np.sqrt(np.sum(emb ** 2, axis=1))
        emb = emb / np.dot(emb_.reshape(-1, 1), np.ones((1, emb.shape[1])))
        emb[0, :] = 0.

        self.emb = theano.shared(
            value=np.asarray(emb, dtype=theano.config.floatX),
            name=prefix + 'emb',
            borrow=True
        )

        self.kbm = T.constant(
            x=kbm,
            name=prefix + 'kbm',
            ndim=2,
            dtype='int32'
        )

        self.kbm_mask = T.constant(
            x=kbm_mask,
            name=prefix + 'kbm_mask',
            ndim=2,
            dtype=theano.config.floatX
        )

        def _random_weights(x_dim, y_dim):
            return np.random.uniform(
                low=-np.sqrt(6. / (x_dim + y_dim)),
                high=np.sqrt(6. / (x_dim + y_dim)),
                size=(x_dim, y_dim)
            ).astype(theano.config.floatX)

        self.gru_W = theano.shared(
            value=np.concatenate(
                [_random_weights(word_size, hidden_size),
                 _random_weights(word_size, hidden_size),
                 _random_weights(word_size, hidden_size)],
                axis=1
            ).astype(theano.config.floatX),
            name=prefix+'gru_W',
            borrow=True
        )

        self.gru_U = theano.shared(
            value=np.concatenate(
                [_random_weights(hidden_size, hidden_size),
                 _random_weights(hidden_size, hidden_size),
                 _random_weights(hidden_size, hidden_size)],
                axis=1
            ).astype(theano.config.floatX),
            name=prefix+'gru_U',
            borrow=True
        )

        self.gru_B = theano.shared(
            value=np.zeros((3 * hidden_size,)).astype(theano.config.floatX),
            name=prefix+'b',
            borrow=True
        )


    def emb_set_value_zero(self):
        self.emb = T.set_subtensor(self.emb[0:], 0.)