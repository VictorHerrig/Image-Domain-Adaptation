import tensorflow as tf

def train_input_fn(x_labelled, x_unlabelled, l_labels, u_labels, batch_size, buffer_size, epochs,
                   use_val = False, x_lab_val=None, x_unlab_val=None, y_lab_val=None, y_unlab_val=None):
    '''
    Function for creating the input iterator for training the model
    :param x_labelled: Labelled training images
    :param x_unlabelled: Unlabelled training images
    :param l_labels: Labels for labelled training set
    :param u_labels: Labels for 'unlabelled' training set
    :param batch_size: Batch size to use for the iterator
    :param buffer_size: Buffer size to use when shuffling the data
    :param epochs: Number of epochs to repeat for - value of None repeats indefinitely
    :param use_val: Whether or not to use a validation set
    :param x_lab_val: Labelled validation images
    :param x_unlab_val: Unlabelled validation images
    :param y_lab_val: Labels for the labelled validation images
    :param y_unlab_val: Labels for the 'unlabelled' validation images
    :return: Dictionary containing iterator ops fr initialization and iterating
    '''
    # TODO: Different batch size for validation data

    dataset = tf.data.Dataset.from_tensor_slices((x_labelled, x_unlabelled, l_labels, u_labels))
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, epochs))
    dataset = dataset.batch(batch_size).prefetch(1)
    iter = dataset.make_initializable_iterator()
    xlb, xub, ylb, yub = iter.get_next()
    iter_init = iter.initializer

    ret = {
        'iter_init': iter_init,
        'x_labelled': xlb,
        'x_unlabelled': xub,
        'labels': {
            'labelled': ylb,
            'unlabelled': yub
        }
    }

    if use_val:
        v_d = tf.data.Dataset.from_tensor_slices((x_lab_val, x_unlab_val, y_lab_val, y_unlab_val))
        v_d = v_d.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, None))
        v_d = v_d.batch(batch_size).prefetch(1)
        v_iter = v_d.make_initializable_iterator()
        vxl, vxu, vyl, vyu = v_iter.get_next()
        v_iter_init = v_iter.initializer
        ret['validation'] = {
            'iter_init': v_iter_init,
            'x_labelled': vxl,
            'x_unlabelled': vxu,
            'labels': {
                'labelled': vyl,
                'unlabelled': vyu
            }
        }

    return ret

def test_input_fn(x_test_labelled, x_test_unlabelled, labelled_labels, unlabelled_labels, batch_size, buffer_size):
    '''
    Function for creating the input iterator for testing the model
    :param x_test_labelled: Labelled test images
    :param x_test_unlabelled: Unlabelled test images
    :param labelled_labels: Labels for labelled test images
    :param unlabelled_labels: Labels for 'unlabelled' test images
    :param batch_size: Batch size to use for the iterator
    :param buffer_size: Buffer size to use when shuffling the data
    :return:
    '''
    dataset = tf.data.Dataset.from_tensor_slices((x_test_labelled, x_test_unlabelled,
                                                  labelled_labels, unlabelled_labels))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size).prefetch(1)
    iter = dataset.make_initializable_iterator()
    x_l_b, x_u_b, y_l_b, y_u_b = iter.get_next()
    iter_init = iter.initializer

    return {
        'iter_init': iter_init,
        'x_labelled': x_l_b,
        'x_unlabelled': x_u_b,
        'labels': {
            'labelled': y_l_b,
            'unlabelled': y_u_b
        }
    }

def predict_input_fn(x):
    dataset = tf.data.Dataset.from_tensor_slices((x))
    dataset = dataset.prefetch(1)
    iter = dataset.make_initializable_iterator()
    x_b = iter.get_next()
    iter_init = iter.initializer

    return{
        'iter_init': iter_init,
        'x_labelled': None,
        'x_unlabelled': x_b,
        'labels': None
    }