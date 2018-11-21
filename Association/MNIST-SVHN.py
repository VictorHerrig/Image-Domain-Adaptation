import numpy as np
import argparse
import tensorflow as tf
from train import train
from test import test
from input import load_mnist, load_svhn


def prepare_inut(params):
    print('Preparing data')
    x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled = load_mnist(60000, 10000)
    x_train_labelled, y_train_labelled, x_test_labelled, y_test_labelled = \
        load_svhn('svhn_train.mat', 'svhn_test.mat', 60000, 10000)

    params['target_shape'] = (x_test_labelled.shape[1], x_test_labelled.shape[2])

    return x_train_labelled, y_train_labelled, x_train_unlabelled, y_train_unlabelled, params,\
           x_test_labelled, y_test_labelled, x_test_unlabelled, y_test_unlabelled

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model params')
    parser.add_argument('--model_dir', help='Directory in which to store the model', type=str, default='model')
    parser.add_argument('--epochs', help='Number of epochs over which to train', type=int, default=None)
    parser.add_argument('--steps', help='Number of steps over which to train', type=int, default=10000)
    parser.add_argument('--lr', help='Learning rate of the optimizer', type=float, default=0.0001)
    parser.add_argument('--min_lr', help='Minimum learning rate to use', type=float, default=0.000001)
    parser.add_argument('--lr_decay_steps', help='Number of steps for lr decay', type=str, default=10000)
    parser.add_argument('--lr_decay_rate', help='Rate of lr decay', type=float, default=0.33)
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
    parser.add_argument('--model_to_use', help='The model to use for embedding - default, Adam, Inception or Dense', type=str, default='')
    args = parser.parse_args()

    #Exactly one of epochs or steps is None
    assert ((args.epochs is None or args.steps is None) and not (args.epochs is None and args.steps is None))

    params = {
        'assoc_w': args.assoc_w,
        'lr': args.lr,
        'min_lr': args.min_lr,
        'lr_decay_rate': args.lr_decay_rate,
        'lr_decay_steps': args.lr_decay_steps,
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
        'conv_dropout': args.conv_dropout,
        'fc_dropout': args.fc_dropout,
        'l2_mag': args.l2_mag,
        'use_val': args.use_val,
        'model_to_use': args.model_to_use
    }

    if args.train or args.test:
        data_in = prepare_inut(params)

        if args.train:
            with tf.Graph().as_default():
                # train(data_in[0], data_in[1], data_in[2], data_in[3],
                #       data_in[4], data_in[5], data_in[6], data_in[7], data_in[8])
                train(*data_in)
        if args.test:
            with tf.Graph().as_default():
                test(data_in[5], data_in[6], data_in[7], data_in[8], data_in[4])
