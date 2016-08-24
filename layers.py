import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

class LogisticRegression(object):
    def __init__(self, x, y, in_size, out_size, prefix='lr_'):

        self.W = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(6. / (in_size + out_size)),
                high=np.sqrt(6. / (in_size + out_size)),
                size=(in_size, out_size)
            ).astype(theano.config.floatX),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(6. / (in_size + out_size)),
                high=np.sqrt(6. / (in_size + out_size)),
                size=(out_size,)
            ).astype(theano.config.floatX),
            name='b',
            borrow=True
        )

        self.y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)

        self.y_d = T.argmax(self.y_given_x, axis=1)

        self.loss = -T.mean(T.log(self.y_given_x)[T.arange(y.shape[0]), y])

        self.error = T.mean(T.neq(self.y_d, y))

        self.params = {prefix+'W':self.W, prefix+'b':self.b}

class Embedding_layer(object):
    def __init__(self, x, emb, word_size=100, prefix='embedd_layer_'):
        n_steps = x.shape[1]
        n_samples = x.shape[0]

        self.x = T.transpose(x)

        self.emb = theano.shared(
            value=np.asarray(emb, dtype=theano.config.floatX),
            name=prefix + 'emb',
            borrow=True
        )

        self.output = self.emb[self.x.flatten()].reshape([n_steps, n_samples, word_size])

        self.params = {prefix+'emb':self.emb}

class LSTM_layer(object):
    def __init__(self, x, mask=None, in_size=100, hidden_size=400, mean_pooling=True, prefix='lstm_'):
        """attention, every column in input is a sample"""
        def random_weights(x_dim, y_dim):
            return np.random.uniform(
                low=-np.sqrt(6. / (x_dim + y_dim)),
                high=np.sqrt(6. / (x_dim + y_dim)),
                size=(x_dim, y_dim)
            ).astype(theano.config.floatX)

        self.W = theano.shared(
            value=np.concatenate(
                [random_weights(in_size, hidden_size),
                 random_weights(in_size, hidden_size),
                 random_weights(in_size, hidden_size),
                 random_weights(in_size, hidden_size)],
                axis=1
            ).astype(theano.config.floatX),
            name=prefix+'W',
            borrow=True
        )

        self.U = theano.shared(
            value=np.concatenate(
                [random_weights(hidden_size, hidden_size),
                 random_weights(hidden_size, hidden_size),
                 random_weights(hidden_size, hidden_size),
                 random_weights(hidden_size, hidden_size)],
                axis=1
            ).astype(theano.config.floatX),
            name=prefix+'U',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros((4 * hidden_size,)).astype(theano.config.floatX),
            name=prefix+'b',
            borrow=True
        )

        assert mask is not None

        n_steps = x.shape[0]
        if x.ndim == 3:
            n_samples = x.shape[1]
        else:
            n_samples = 1

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_):
            preact = T.dot(h_, self.U)
            preact += x_

            i = T.nnet.sigmoid(_slice(preact, 0, hidden_size))
            f = T.nnet.sigmoid(_slice(preact, 1, hidden_size))
            o = T.nnet.sigmoid(_slice(preact, 2, hidden_size))
            c = T.tanh(_slice(preact, 3, hidden_size))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * T.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

        input = (T.dot(x, self.W) + self.b)

        rval, updates = theano.scan(_step,
                                    sequences=[mask, input],
                                    outputs_info=[T.alloc(numpy_floatX(0.), n_samples, hidden_size),
                                                  T.alloc(numpy_floatX(0.), n_samples, hidden_size)],
                                    name=prefix+'_scan',
                                    n_steps=n_steps)

        if mean_pooling:
            hidden_sum = (rval[0] * mask[:, :, None]).sum(axis=0)
            self.output = hidden_sum / mask.sum(axis=0)[:, None]
        else:
            self.output = rval[0][-1, :, :]
            self.out_all = rval[0]

        self.params  = {prefix+'W' : self.W, prefix+'U': self.U, prefix+'b': self.b}

class Dropout_layer(object):
    def __init__(self, input):
        use_noise = theano.shared(numpy_floatX(0.))
        trng = RandomStreams(415)

        self.output = T.switch(use_noise,
                               (input * trng.binomial(input.shape, p=0.5, n=1, dtype=input.dtype)),
                               input * 0.5
                               )