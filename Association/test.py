import tensorflow as tf
from model import model_fn
from input import test_input_fn
import os
from tensorflow.python.tools import inspect_checkpoint as chkp

def test_set(x_l_val, x_u_val, y_l_val, y_u_val, params):
    x_l = tf.placeholder(x_l_val.dtype, x_l_val.shape)
    y_l = tf.placeholder(y_l_val.dtype, y_l_val.shape)
    x_u = tf.placeholder(x_u_val.dtype, x_u_val.shape)
    y_u = tf.placeholder(y_u_val.dtype, y_u_val.shape)
    inputs = test_input_fn(x_l,
                           x_u,
                           y_l,
                           y_u,
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
    num = y_l_val.shape[0]
    step = 0
    saver = tf.train.Saver()
    chkp.print_tensors_in_checkpoint_file(os.path.join(params['model_dir'], 'model.ckpt'), tensor_name='', all_tensors=False, all_tensor_names=True)

    with tf.Session() as sess:
        tf.logging.info('Starting testing')
        sess.run(model['var_init'])
        sess.run(acc_init)
        saver.restore(sess, os.path.join(params['model_dir'], 'model.ckpt'))
        # tf.saved_model.loader.load(sess, None, os.path.join(params['model_dir'], 'saved_model/'))
        sess.run(model['iter_init'], {x_l: x_l_val, x_u: x_u_val, y_l: y_l_val, y_u: y_u_val})
        writer = tf.summary.FileWriter(os.path.join(params['model_dir'], 'test_summaries'), sess.graph)
        while True:
            try:
                step += 1
                _, _, l_acc_val, u_acc_val = \
                    sess.run([update_l_acc, update_u_acc, l_accuracy, u_accuracy])
                total_l_acc = l_acc_val
                total_u_acc = u_acc_val
                if params['log_step']:
                    if step % params['log_step'] == 0:
                        tf.logging.info('Loss: {}; Labelled Accuracy: {}; Unlabelled Accuracy: {}'.format(loss_sum / step, total_l_acc, total_u_acc))
            except tf.errors.OutOfRangeError:
                break
    return total_l_acc, total_u_acc

def test(x_test_unlabelled,
         unlabelled_labels,
         x_test_labelled,
         labelled_labels,
         params):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting testing')
    params['reshape'] = 'u'
    params['reuse'] = False
    l_acc, u_acc = test_set(x_test_labelled, x_test_unlabelled, labelled_labels, unlabelled_labels, params)
    tf.logging.info('MEAN LABELLED ACCURACY: {}'.format(l_acc))
    tf.logging.info('MEAN UNLABELLED ACCURACY: {}'.format(u_acc))
    tf.logging.info('Finished testing')