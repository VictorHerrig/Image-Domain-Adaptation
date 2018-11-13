import tensorflow as tf
from model import model_fn
from input import train_input_fn
import os
from tensorflow.python.tools import inspect_checkpoint as chkp

def train_step(sess, train_op, loss, summaries, global_step, writer, params, saver):
    if params['log_step'] is not None:
        _, loss_val, sumr, global_step_val = sess.run([train_op, loss, summaries, global_step])
        writer.add_summary(sumr, global_step_val)
        train_step.loss_sum += loss_val
        train_step.steps += 1
        # log and save on specified step increments
        if global_step_val % params['log_step'] == 0 or global_step_val == params['steps']:
            tf.logging.info('step: {}; loss: {}'.format(global_step_val, train_step.loss_sum / train_step.steps))
            save_dir = os.path.join(params['model_dir'], 'model.ckpt')
            saver.save(sess, save_dir)
            train_step.loss_sum = 0
            train_step.steps = 0
        if global_step_val == params['steps'] or global_step_val == params['steps'] - 1:
            chkp.print_tensors_in_checkpoint_file(os.path.join(params['model_dir'], 'model.ckpt'), tensor_name='',
                                                  all_tensors=False, all_tensor_names=True)
    else:
        _, sumr, global_step_val = sess.run([train_op, summaries, global_step])
        writer.add_summary(sumr, global_step_val)
train_step.loss_sum = 0
train_step.steps = 0

def train(x_train_labelled,
          x_train_unlabelled,
          train_labels,
          params):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Setting up training')

    xl = tf.placeholder(x_train_labelled.dtype, x_train_labelled.shape)
    xu = tf.placeholder(x_train_unlabelled.dtype, x_train_unlabelled.shape)
    y = tf.placeholder(train_labels.dtype, train_labels.shape)

    inputs = train_input_fn(xl,
                            xu,
                            y,
                            params['batch_size'],
                            params['buffer_size'],
                            params['epochs'])
    model = model_fn(inputs, tf.estimator.ModeKeys.TRAIN, params)
    loss = model['loss']
    train_op = model['train_op']
    summaries = model['summaries']
    global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        tf.logging.info('Starting training')
        sess.run(model['var_init'])
        writer = tf.summary.FileWriter(os.path.join(params['model_dir'], 'train_summaries'), sess.graph)

        sess.run(model['iter_init'], {xl: x_train_labelled, xu:x_train_unlabelled, y:train_labels})

        # Count by epochs
        if params['steps'] is None:
            tf.logging.info('Counting by epochs')
            while True:
                try:
                    train_step(sess, train_op, loss, summaries, global_step, writer, params, saver)
                except:
                    break
        #Count by steps
        else:
            tf.logging.info('Counting by steps')
            for s in range(params['steps']):
                train_step(sess, train_op, loss, summaries, global_step, writer, params, saver)
        # tf.saved_model.simple_save(sess, os.path.join(params['model_dir'], 'saved_model/'))
    tf.logging.info('Finished training')