import theano
import theano.tensor as T
import numpy as np

import config
import optimizers
from reader import get_embedding_matrix_from_param_file
from reader import gkhmc_iterator
from config import options

def run():
    x = T.imatrix('x')
    y = T.ivector('y')
    mask = T.matrix('mask', dtype=theano.config.floatX)
    lr = T.scalar(name='lr')

    np_emb = get_embedding_matrix_from_param_file(config.embedding_param_file)

    print '...building model'
    model = options['model'](x, y, mask, np_emb)

    cost = model.loss
    grads = T.grad(cost, wrt=list(model.params.values()))
    optimizer = optimizers.rmsprop
    f_grad_shared, f_update = optimizer(lr, model.params, grads, x, mask, y, cost)

    detector = theano.function(inputs=[x, mask, y], outputs=model.error)

    valid_freq = 5

    print '...training model'
    for i in xrange(options['max_epochs']):
        total_loss = 0.
        idx = 0
        for x_, mask_, y_ in gkhmc_iterator(path='data/GKHMC.pickle', batch_size=options['batch_size']):
            this_cost = f_grad_shared(x_, mask_, y_)
            f_update(options['lrate'])
            total_loss += this_cost
            print '\r', 'epoch:', i, ', idx:', idx, ', this_loss:', this_cost,
            idx += 1
        print ', total loss:', total_loss

        if (i + 1) % valid_freq == 0:
            errors = []
            for x_, mask_, y_ in gkhmc_iterator(path='data/GKHMC.pickle', batch_size=options['batch_size'], is_train=True):
                error = detector(x_, mask_, y_)
                errors.append(error)
            print '\ttrain error of epoch ' + str(i) + ': ' + str(np.mean(errors) * 100) + '%'

            errors = []
            for x_, mask_, y_ in gkhmc_iterator(path='../data/imdb.pkl', batch_size=128, is_train=False):
                error = detector(x_, mask_, y_)
                errors.append(error)
            print '\ttest error of epoch ' + str(i) + ': ' + str(np.mean(errors) * 100) + '%'


if __name__ == '__main__':
    run()