import tensorflow as tf
from model import model_fn
from input import test_input_fn
import os
from tensorflow.python.tools import inspect_checkpoint as chkp

'''
def test_set(x_test_labelled,
             x_test_unlabelled,
             y_test_labelled,
             y_test_unlabelled,
             params):
    # Prepare input and model
    x_l = tf.placeholder(x_test_labelled.dtype, x_test_labelled.shape)
    y_l = tf.placeholder(y_test_labelled.dtype, y_test_labelled.shape)
    x_u = tf.placeholder(x_test_unlabelled.dtype, x_test_unlabelled.shape)
    y_u = tf.placeholder(y_test_unlabelled.dtype, y_test_unlabelled.shape)
    inputs = test_input_fn(x_l,x_u,y_l,y_u,
                           params['batch_size'],
                           params['buffer_size'])
    model = model_fn(inputs, tf.estimator.ModeKeys.EVAL, params)
    l_accuracy = model['l_accuracy']
    u_accuracy = model['u_accuracy']
    update_l_acc = model['update_l_acc']
    update_u_acc = model['update_u_acc']
    acc_init = model['acc_init']
    loss_sum = 0.0
    total_l_acc = 0.0
    total_u_acc = 0.0
    step = 0
    saver = tf.train.Saver()
    #chkp.print_tensors_in_checkpoint_file(os.path.join(params['model_dir'], 'model.ckpt'), tensor_name='', all_tensors=False, all_tensor_names=True)

    with tf.Session() as sess:
        tf.logging.info('Starting testing')
        sess.run(model['var_init'])
        sess.run(acc_init)
        # Restoring trained weights from training in params['model_dir']
        saver.restore(sess, os.path.join(params['model_dir'], 'model.ckpt'))
        sess.run(model['iter_init'], {x_l: x_test_labelled, x_u: x_test_unlabelled, y_l: y_test_labelled, y_u: y_test_unlabelled})
        #writer = tf.summary.FileWriter(os.path.join(params['model_dir'], 'test_summaries'), sess.graph)
        while True:
            try:
                step += 1
                # Update accuracy values
                _, _, l_acc_val, u_acc_val = \
                    sess.run([update_l_acc, update_u_acc, l_accuracy, u_accuracy])
                total_l_acc = l_acc_val
                total_u_acc = u_acc_val
                if params['log_step']:
                    if step % params['log_step'] == 0:
                        # Log accuracies so far if appropriate
                        tf.logging.info('Loss: {}; Labelled Accuracy: {}; Unlabelled Accuracy: {}'.format(loss_sum / step, total_l_acc, total_u_acc))
            except tf.errors.OutOfRangeError:
                # At the end of the dataset
                break
    return total_l_acc, total_u_acc
'''

def test(x_test_labelled,
         y_test_labelled,
         x_test_unlabelled,
         y_test_unlabelled,
         params):
    '''
    #Function for training the association model
    :param x_test_labelled: Labelled test images
    :param x_test_unlabelled: Unlabelled test images
    :param y_test_labelled: Labels for labelled test images
    :param y_test_unlabelled: Labels for 'unlabelled' test images
    :param params: Dict of parameters
    :return: No value, prints the average accuracies
    '''

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting testing')
    params['reshape'] = 'u'
    params['reuse'] = False
    params['use_val'] = False
    # l_acc, u_acc = test_set(x_test_labelled, x_test_unlabelled, y_test_labelled, y_test_unlabelled, params)

    # Prepare input and model
    x_l = tf.placeholder(x_test_labelled.dtype, x_test_labelled.shape)
    y_l = tf.placeholder(y_test_labelled.dtype, y_test_labelled.shape)
    x_u = tf.placeholder(x_test_unlabelled.dtype, x_test_unlabelled.shape)
    y_u = tf.placeholder(y_test_unlabelled.dtype, y_test_unlabelled.shape)
    inputs = test_input_fn(x_l, y_l, x_u, y_u,
                           params['batch_size'],
                           params['buffer_size'])
    model = model_fn(inputs, tf.estimator.ModeKeys.EVAL, params)
    l_accuracy = model['l_accuracy']
    u_accuracy = model['u_accuracy']
    update_l_acc = model['update_l_acc']
    update_u_acc = model['update_u_acc']
    acc_init = model['acc_init']
    total_l_acc, total_u_acc, step = 0.0, 0.0, 0
    saver = tf.train.Saver()
    # chkp.print_tensors_in_checkpoint_file(os.path.join(params['model_dir'], 'model.ckpt'), tensor_name='', all_tensors=False, all_tensor_names=True)

    with tf.Session() as sess:
        tf.logging.info('Starting testing')
        sess.run(model['var_init'])
        sess.run(acc_init)
        # Restoring trained weights from training
        saver.restore(sess, os.path.join(params['model_dir'], 'model.ckpt'))
        sess.run(model['iter_init'],
                 {x_l: x_test_labelled, y_l: y_test_labelled, x_u: x_test_unlabelled, y_u: y_test_unlabelled})
        # writer = tf.summary.FileWriter(os.path.join(params['model_dir'], 'test_summaries'), sess.graph)
        while True:
            try:
                step += 1
                # Update accuracy values
                _, _, l_acc_val, u_acc_val = \
                    sess.run([update_l_acc, update_u_acc, l_accuracy, u_accuracy])
                total_l_acc = l_acc_val
                total_u_acc = u_acc_val
                if params['log_step']:
                    if step % params['log_step'] == 0:
                        # Log accuracies so far if appropriate
                        tf.logging.info(
                            'Labelled Accuracy: {}; Unlabelled Accuracy: {}'.format(total_l_acc, total_u_acc))
            except tf.errors.OutOfRangeError:
                # At the end of the dataset
                break

    tf.logging.info('MEAN LABELLED ACCURACY: {}'.format(total_l_acc))
    tf.logging.info('MEAN UNLABELLED ACCURACY: {}'.format(total_u_acc))
    tf.logging.info('Finished testing')