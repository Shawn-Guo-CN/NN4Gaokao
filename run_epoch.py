import theano
import theano.tensor as T
import numpy as np

import config
import optimizers
from reader import get_embedding_matrix_from_param_file
from reader import gkhmc_iterator
from config import options

def pred_check(p_ds, ys):
    right_num = 0
    total_num = 0
    for i in xrange(len(p_ds) / 4):
        y_local = ys[4 * i: 4 * (i + 1)]
        p_local = p_ds[4 * i: 4 * (i + 1)]
        total_num += 1
        right_index = y_local.index(1)
        if right_index == p_local.index(max(p_local)):
            right_num += 1
    return right_num, total_num

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
    optimizer = options['optimizer']
    f_grad_shared, f_update = optimizer(lr, model.params, grads, x, mask, y, cost)

    detector = theano.function(inputs=[x, mask, y], outputs=model.error)
    p_predictor = theano.function(inputs=[x, mask], outputs=model.p_d)

    p_ds = []
    ys = []
    for x_, mask_, y_ in gkhmc_iterator(path='data/GKHMC.pickle', batch_size=options['valid_batch_size'],
                                        is_train=False):
        p_d = p_predictor(x_, mask_)
        p_ds.extend(p_d)
        ys.extend(y_)
    right_num , total_num = pred_check(p_ds, ys)
    print right_num, '/', total_num

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

        if (i + 1) % options['valid_freq'] == 0:
            errors = []
            for x_, mask_, y_ in gkhmc_iterator(path='data/GKHMC.pickle', batch_size=options['valid_batch_size'],
                                                is_train=True):
                error = detector(x_, mask_, y_)
                errors.append(error)
            print '\ttrain error of epoch ' + str(i) + ': ' + str(np.mean(errors) * 100) + '%'

            p_ds = []
            ys = []
            for x_, mask_, y_ in gkhmc_iterator(path='data/GKHMC.pickle', batch_size=options['valid_batch_size'],
                                                is_train=False):
                p_d = p_predictor(x_, mask_)
                p_ds.extend(p_d)
                ys.extend(y_)
            right_num, total_num = pred_check(p_ds, ys)
            print '\ttest performance of epoch', i, ':', right_num, '/', total_num, '\t', \
                float(right_num) / float(total_num), '%'


if __name__ == '__main__':
    run()