import numpy as np
import argparse
import tensorflow.keras.datasets.mnist as mnist
import scipy.io as spio
from train import train
from test import test
import tensorflow as tf

def reshape_mnist(set):
    image = np.true_divide(set, 255)
    image = np.expand_dims(image, -1)
    return image

def reshape_svhn(set):
    ret = set['X']
    ret = np.true_divide(ret, 255)
    ret = np.transpose(ret, [3,0,1,2])
    return ret

def prepare_inut(params):
    print('Preparing data')
    (x_train_unlabelled, mnist_train_labels), (x_test_unlabelled, test_u_labels) = mnist.load_data()
    x_train_unlabelled = reshape_mnist(x_train_unlabelled)
    x_test_unlabelled = reshape_mnist(x_test_unlabelled)

    svhn_train = spio.loadmat('svhn_train.mat')
    x_train_labelled = reshape_svhn(svhn_train)[:60000, :, :, :]
    svhn_train_labels = svhn_train['y'][:60000, :]
    # 0s have label of 10 in SVHN... Fix that
    svhn_train_labels = np.apply_along_axis(lambda x: x if not np.equal(x, 10) else 0, -1, svhn_train_labels)
    mnist_train_labels = np.reshape(mnist_train_labels, (mnist_train_labels.shape[0]))
    svhn_train_labels = np.reshape(svhn_train_labels, (svhn_train_labels.shape[0]))
    params['target_shape'] = (x_train_labelled.shape[1], x_train_labelled.shape[2])

    svhn_test = spio.loadmat('svhn_test.mat')
    x_test_labelled = reshape_svhn(svhn_test)[:10000, :, :, :]
    test_svhn_labels = svhn_test['y'][:10000, :]
    # 0s have label of 10 in SVHN... Fix that
    test_svhn_labels = np.apply_along_axis(lambda x: x if not np.equal(x, 10) else 0, -1, test_svhn_labels)
    test_l_labels = np.reshape(test_svhn_labels, (test_svhn_labels.shape[0]))
    params['target_shape'] = (x_test_labelled.shape[1], x_test_labelled.shape[2])
    return x_train_labelled, x_train_unlabelled, svhn_train_labels, mnist_train_labels, params,\
           x_test_labelled, x_test_unlabelled, test_l_labels, test_u_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model params')
    parser.add_argument('--model_dir', help='Directory in which to store the model', type=str, default='model')
    parser.add_argument('--epochs', help='Number of epochs over which to train', type=int, default=None)
    parser.add_argument('--steps', help='Number of steps over which to train', type=int, default=10000)
    parser.add_argument('--lr', help='Learning rate of the optimizer', type=float, default=0.0001)
    parser.add_argument('--visit_w', help='Weight assigned to the visit loss', type=float, default=1.0)
    parser.add_argument('--assoc_w', help='Weight assigned to the association loss', type=float, default=1.0)
    parser.add_argument('--walk_w', help='Weight assigned to the walker loss', type=float, default=1.0)
    parser.add_argument('--batch_size', help='Batch size to use for training', type=int, default=100)
    parser.add_argument('--buffer_size', help='Buffer size to use when shuffling the dataset', type=int, default=1000)
    parser.add_argument("--log_step", help="Log each step", type=int, default=None)
    parser.add_argument('--train', help='Whether or not to train the model', action='store_true', default=False)
    parser.add_argument('--test', help='Whether or not to test the model', action='store_true', default=False)
    parser.add_argument('--predict', help='Whether or not to predict', action='store_true', default=False)
    # parser.add_argument('--assoc_thresh', help='Number of steps before adding assoc loss', type=int, default=0)
    parser.add_argument('--fc_dropout', help='Dropout rate for the fc layer', type=float, default=0.0)
    parser.add_argument('--conv_dropout', help='Dropout rate for the conv blocks', type=float, default=0.0)
    parser.add_argument('--l2_mag', help='Lambda value for l2 regularization', type=float, default=0.0)
    parser.add_argument('--use_val', help='Whether or not to use validation', action='store_true', default=False)
    args = parser.parse_args()

    #Exactly one of epochs or steps is None
    assert ((args.epochs is None or args.steps is None) and not (args.epochs is None and args.steps is None))

    params = {
        'assoc_w': args.assoc_w,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'w_w': args.walk_w,
        'v_w': args.visit_w,
        'num_classes': 10,
        'model_dir': args.model_dir,
        'buffer_size': args.buffer_size,
        'epochs': args.epochs,
        'steps': args.steps,
        'log_step': args.log_step,
        'reshape': 'u',
        # 'assoc_thresh': args.assoc_thresh,
        'conv_dropout': args.conv_dropout,
        'fc_dropout': args.fc_dropout,
        'l2_mag': args.l2_mag,
        'use_val': args.use_val
    }

    if args.test and args.train:
        data_in = prepare_inut(params)
        with tf.Graph().as_default():
            train(data_in[0], data_in[1], data_in[2], data_in[3],
                  data_in[4], data_in[5], data_in[6], data_in[7], data_in[8])
        with tf.Graph().as_default():
            test(data_in[6], data_in[8], data_in[5], data_in[7], data_in[4])

    elif args.train:
        data_in = prepare_inut(params)
        train(data_in[0], data_in[1], data_in[2], data_in[3],
              data_in[4], data_in[5], data_in[6], data_in[7], data_in[8])

    elif args.test:
        print('Preparing testing data')
        (_, _), (x_test_unlabelled, test_u_labels) = mnist.load_data()
        x_test_unlabelled = reshape_mnist(x_test_unlabelled)

        svhn_test = spio.loadmat('svhn_test.mat')
        x_test_labelled = reshape_svhn(svhn_test)[:10000, :, :, :]
        test_svhn_labels = svhn_test['y'][:10000,:]
        # 0s have label of 10 in SVHN... Fix that
        test_svhn_labels = np.apply_along_axis(lambda x: x if not np.equal(x, 10) else 0, -1, test_svhn_labels)
        test_l_labels = np.reshape(test_svhn_labels, (test_svhn_labels.shape[0]))
        params['target_shape'] = (x_test_labelled.shape[1], x_test_labelled.shape[2])
        test(x_test_unlabelled, test_u_labels, x_test_labelled, test_l_labels, params)
