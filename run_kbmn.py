import theano
import theano.tensor as T
import numpy as np
import cPickle

import config
from reader import get_embedding_matrix_from_param_file
from reader import gkhmc_qla_iterator
from reader import get_kb_from_param_file
from config import options
from models import KBMN

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

def run_epoch():
    # define symbolic variables
    q = T.imatrix('q')
    q_mask = T.matrix('q_mask', dtype=theano.config.floatX)
    l = T.imatrix('l')
    l_mask = T.matrix('l_mask', dtype=theano.config.floatX)
    a = T.imatrix('a')
    a_mask = T.matrix('a_mask', dtype=theano.config.floatX)
    y = T.ivector('y')
    lr = T.scalar(name='lr')

    np_emb = get_embedding_matrix_from_param_file(config.embedding_param_file)
    np_kbm, np_kbm_mask = get_kb_from_param_file(config.kb_param_file)

    # build model
    print '...building model'
    model = KBMN(q, q_mask, l, l_mask, a, a_mask, y, np_kbm, np_kbm_mask, np_emb, options['word_size'],
                 options['hidden_size'], options['use_dropout'], options['drop_p'])

    cost = model.loss
    grads = T.grad(cost, wrt=list(model.params.values()))
    optimizer = options['optimizer']
    f_grad_shared, f_update = optimizer(lr, model.params, grads, [q, q_mask, l, l_mask, a, a_mask, y], cost)

    detector = theano.function(
        inputs=[q, q_mask, l, l_mask, a, a_mask, y],
        outputs=model.error,
        on_unused_input='ignore'
    )
    p_predictor = theano.function(
        inputs=[q, q_mask, l, l_mask, a, a_mask],
        outputs=model.p_d,
        on_unused_input='ignore'
    )

    # load parameters from specified file
    if not options['loaded_params'] is None:
        print '...loading parameters from ' + options['loaded_params']
        file_name = options['loaded_params']
        with open(file_name, 'rb') as f:
            param_dict = cPickle.load(f)
            for k, v in model.params.items():
                v.set_value(param_dict[k])

    # test the performance of initialized parameters
    print '...testing the performance of initialized parameters'
    p_ds = []
    ys = []
    for q_, q_mask_, l_, l_mask_, a_, a_mask_, y_ in gkhmc_qla_iterator(
            path=config.dataset, batch_size=options['valid_batch_size'], is_train=False):
        p_d = p_predictor(q_, q_mask_, l_, l_mask_, a_, a_mask_)
        p_ds.extend(p_d)
        ys.extend(y_)
    right_num, total_num = pred_check(p_ds, ys)
    print right_num, '/', total_num

    best_perform = -np.inf

    print '...training model'
    for i in xrange(options['max_epochs']):
        total_loss = 0.
        idx = 0
        for q_, q_mask_, l_, l_mask_, a_, a_mask_, y_ in gkhmc_qla_iterator(
                path=config.dataset, batch_size=options['batch_size'], is_train=True):
            model.emb_set_value_zero()
            this_cost = f_grad_shared(q_, q_mask_, l_, l_mask_, a_, a_mask_, y_)
            f_update(options['lrate'])
            total_loss += this_cost
            print '\r', 'epoch:', i, ', idx:', idx, ', this_loss:', this_cost,
            idx += 1
        print ', total loss:', total_loss

        # validate model performance when necessary
        if (i + 1) % options['valid_freq'] == 0:
            # test performance on train set
            errors = []
            for q_, q_mask_, l_, l_mask_, a_, a_mask_, y_ in gkhmc_qla_iterator(
                    path=config.dataset, batch_size=options['valid_batch_size'], is_train=True):
                error = detector(q_, q_mask_, l_, l_mask_, a_, a_mask_, y_)
                errors.append(error)
            print '\ttrain error of epoch ' + str(i) + ': ' + str(np.mean(errors) * 100) + '%'

            # test performance on test set
            p_ds = []
            ys = []
            for q_, q_mask_, l_, l_mask_, a_, a_mask_, y_ in gkhmc_qla_iterator(
                    path=config.dataset, batch_size=options['valid_batch_size'], is_train=False):
                p_d = p_predictor(q_, q_mask_, l_, l_mask_, a_, a_mask_)
                p_ds.extend(p_d)
                ys.extend(y_)
            right_num, total_num = pred_check(p_ds, ys)

            # judge whether it's necessary to save the parameters
            save = False
            if float(right_num) / float(total_num) > best_perform:
                best_perform = float(right_num) / float(total_num)
                save = True

            print '\ttest performance of epoch', i, ':', right_num, '/', total_num, '\t', \
                float(right_num * 10000 / total_num) / 100., '%', '\tbest through:', float(int(best_perform * 10000)) / 100.

            # save parameters if need
            if save:
                print '\t...saving parameters'
                file_name = options['param_path'] + model.name + '_hidden' + str(options['hidden_size']) + '_lrate' + \
                            str(options['lrate']) + '_batch' + str(options['batch_size']) + '_epoch' + str(i+1) + \
                            '_perform' + str(float(int(best_perform * 10000)) / 100.) + '.pickle'
                with open(file_name, 'wb') as f:
                    new_dict = {}
                    for k, v in model.params.items():
                        new_dict[k] = v.get_value()
                    cPickle.dump(new_dict, f)


if __name__ == '__main__':
    run_epoch()
