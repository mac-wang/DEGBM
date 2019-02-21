import os
import pickle
import numpy as np
import tensorflow as tf
from model import BiWGAN
from utils import DataInput, calc_auroc, calc_metric, create_logdir, _split_dataset

base_dir = os.path.dirname(os.path.realpath(__file__))
mode = 'talkingdata'
test_batch_size = 1024
method = 'fm'  # or 'cross-e'
weight = 0.9
degree = 1
logdir = create_logdir(mode, method, weight, degree)
save_path = os.path.join(base_dir, logdir)
ano_size = 228423

mapping_ratio = {0.0025: 0.0038}
                 # 0.1: 0.1667, 0.15: 0.2647, 0.2: 0.3749,
                 # 0.25: 0.4999, 0.3: 0.6429, 0.35: 0.8077, 0.4: 1.0}

with open('{}_dataset.pkl'.format(mode), 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    count_list = pickle.load(f)

x_test, y_test = test_set
print('test set:', x_test.shape)

norm, norm_label = x_test[: -ano_size], y_test[: -ano_size]  # 342634
ano, ano_label = x_test[-ano_size:], y_test[-ano_size:]   # 228423


def evaluation(sess, model, ratio):
    (sub_ano, sub_ano_label), _ = _split_dataset(ano, ano_label, mapping_ratio[ratio])
    x = np.concatenate((norm, sub_ano), axis=0)
    y = np.concatenate((norm_label, sub_ano_label), axis=0)

    ano_scores = []
    for _, batch_data in DataInput(x, test_batch_size):
        _ano_score = model.eval(sess, batch_data)
        # Extend
        ano_scores += list(_ano_score)
    ano_scores = np.array(ano_scores).reshape((-1, 1))
    # Calculate auc
    auroc = calc_auroc(y, ano_scores)
    print('Anomaly ratio:{:.4f}\tEval_auroc:{:.4f}'.format(ratio, auroc))
    prec, rec, f1 = calc_metric(y, ano_scores)
    print('Prec:{:.4f}\tRec:{:.4f}\tF1:{:.4f}\n'.format(prec, rec, f1))


with tf.Session() as sess:
    model = BiWGAN(count_list, method, weight=weight, degree=degree)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    model.restore(sess, '{}/ckpt'.format(save_path))

    # evaluation
    for key in mapping_ratio.keys():
        evaluation(sess, model, key)
