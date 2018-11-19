import tensorflow as tf
from model import model_fn
from input import train_input_fn
import os
from tensorflow.python.tools import inspect_checkpoint as chkp


def train(x_train_labelled,
          x_train_unlabelled,
          y_train_labelled,
          y_train_unlabelled,  # TODO: Make this optional
          params,  # TODO: Document necessary keys
          x_validation_labelled=None,
          x_validation_unlabelled=None,
          y_validation_labelled=None,
          y_validaiton_unlabelled=None):  # TODO: Make this optional for validation
    '''
    Function for training the association model
    :param x_train_labelled: Labelled training images
    :param x_train_unlabelled: Unlabelled training images
    :param y_train_labelled: Labels for labelled training set
    :param y_train_unlabelled: Labels for 'unlabelled' training set
    :param params: Dict of parameters
    :param x_validation_labelled: Labelled validation images
    :param x_validation_unlabelled: Unlabelled validation images
    :param y_validation_labelled: Labels for the labelled validation images
    :param y_validaiton_unlabelled: Labels for the 'unlabelled' validation images
    :return: No value, saves a model checkpoint to params['model_dir'] along with tensorboard summaries
    '''
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Setting up training')

    # Prepare input and model
    xl = tf.placeholder(x_train_labelled.dtype, x_train_labelled.shape)
    xu = tf.placeholder(x_train_unlabelled.dtype, x_train_unlabelled.shape)
    yl = tf.placeholder(y_train_labelled.dtype, y_train_labelled.shape)
    yu = tf.placeholder(y_train_unlabelled.dtype, y_train_unlabelled.shape)
    # Check if we're using a validation set
    # TODO: validation set for checking overfitting / convergence
    use_val = x_validation_labelled is not None and \
              x_validation_unlabelled is not None and \
              y_validation_labelled is not None and \
              y_validaiton_unlabelled is not None and \
              params['use_val']
    params['use_val'] = use_val
    if use_val:
        vxl = tf.placeholder(x_validation_labelled.dtype, x_validation_labelled.shape)
        vxu = tf.placeholder(x_validation_unlabelled.dtype, x_validation_unlabelled.shape)
        vyl = tf.placeholder(y_validation_labelled.dtype, y_validation_labelled.shape)
        vyu = tf.placeholder(y_validaiton_unlabelled.dtype, y_validaiton_unlabelled.shape)
        inputs = train_input_fn(xl,xu,yl,yu,
                                params['batch_size'],
                                params['buffer_size'],
                                params['epochs'],
                                True,
                                vxl,vxu,vyl,vyu)
    else:
        inputs = train_input_fn(xl,xu,yl,yu,
                                params['batch_size'],
                                params['buffer_size'],
                                params['epochs'])
    model = model_fn(inputs, tf.estimator.ModeKeys.TRAIN, params)
    loss = model['loss']
    train_op = model['train_op']
    summaries = model['summaries']
    acc_init = model['acc_init']
    global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver(max_to_keep=1)

    def train_step(sess, train_op, loss, summaries, global_step, writer, params, saver, steps, loss_sum):
        '''
        Helper function for training the model
        '''
        if params['log_step'] is not None:
            #If it's a logging step
            _, loss_val, sumr, global_step_val = sess.run([train_op, loss, summaries, global_step])
            writer.add_summary(sumr, global_step_val)
            loss_sum += loss_val
            steps += 1
            # log and reset averages
            if global_step_val % params['log_step'] == 0 or global_step_val == params['steps']:
                tf.logging.info('step: {}; loss: {}'.format(global_step_val, loss_sum / steps))
                save_dir = os.path.join(params['model_dir'], 'model.ckpt')
                saver.save(sess, save_dir)
                loss_sum = 0
                steps = 0
            #Save a checkpoint if this is the last step
            if global_step_val == params['steps'] or global_step_val == params['steps'] - 1:
                chkp.print_tensors_in_checkpoint_file(os.path.join(params['model_dir'], 'model.ckpt'), tensor_name='',
                                                      all_tensors=False, all_tensor_names=True)
        else:
            #If it's a non-logging step
            _, sumr, global_step_val = sess.run([train_op, summaries, global_step])
            writer.add_summary(sumr, global_step_val)
        return steps, loss_sum

    #For keeping track of the average for logging
    current_step = 0
    loss_sum = 0

    with tf.Session() as sess:
        tf.logging.info('Starting training')
        sess.run(model['var_init'])
        sess.run(acc_init)
        writer = tf.summary.FileWriter(os.path.join(params['model_dir'], 'train_summaries'), sess.graph)
        #Initialize the input iterator for non-validation input
        sess.run(model['iter_init'],
                 {xl: x_train_labelled,
                  xu:x_train_unlabelled,
                  yl:y_train_labelled,
                  yu:y_train_unlabelled})
        if use_val:
            #Initialize the input iterator for validation input
            sess.run(model['validation']['iter_init'],
                     {vxl: x_validation_labelled,
                      vxu: x_validation_unlabelled,
                      vyl: y_validation_labelled,
                      vyu: y_validaiton_unlabelled})

        # Count by epochs
        if params['steps'] is None:
            tf.logging.info('Counting by epochs')
            while True:
                try:
                    current_step += 1
                    steps, loss_sum = train_step(sess, train_op,
                                                 loss, summaries,
                                                 global_step, writer,
                                                 params, saver,
                                                 current_step, loss_sum)
                except tf.errors.OutOfRangeError:
                    # At the end of the dataset
                    break
        #Count by steps
        else:
            tf.logging.info('Counting by steps')
            for s in range(params['steps']):
                current_step += 1
                steps, loss_sum = train_step(sess, train_op,
                                             loss, summaries,
                                             global_step, writer,
                                             params, saver,
                                             current_step, loss_sum)
    tf.logging.info('Finished training')