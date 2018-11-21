import tensorflow as tf
import numpy as np
import scipy.io as spio
import tensorflow.keras.datasets.mnist as mnist

def load_mnist(train_limit, test_limit):
    def reshape_mnist(set):
        image = np.true_divide(set, 255)
        image = np.expand_dims(image, -1)  # Add a grayscale dimension
        return image

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = reshape_mnist(x_train)[:train_limit, :, :, :]
    y_train = np.reshape(y_train, (y_train.shape[0]))[:train_limit]
    x_test = reshape_mnist(x_test)[:test_limit, :, :, :]
    y_test = np.reshape(y_test, (y_test.shape[0]))[:test_limit]

    return x_train, y_train, x_test, y_test

def load_svhn(train_file, test_file, train_limit, test_limit):
    def helper(file, limit):
        loaded = spio.loadmat(file)
        x = loaded['X']
        x = np.true_divide(x, 255)
        x = np.transpose(x, [3, 0, 1, 2])[:limit, :, :, :]  # Batch first
        y = loaded['y']
        # 0s have label of 10 in SVHN... Fix that, and reshape to rank 1
        y = np.apply_along_axis(lambda x: x if not np.equal(x, 10) else 0, -1, y)
        y = np.reshape(y, (y.shape[0]))[:limit]
        return x, y

    x_train, y_train = helper(train_file, train_limit)
    x_test, y_test = helper(test_file, test_limit)

    return x_train, y_train, x_test, y_test

def train_input_fn(x_labelled, y_labelled, x_unlabelled, y_unlabelled, batch_size, buffer_size, epochs,
                   use_val=False, x_lab_val=None, y_lab_val=None, x_unlab_val=None, y_unlab_val=None):
    '''
    Function for creating the input iterator for training the model
    :param x_labelled: Labelled training images
    :param x_unlabelled: Unlabelled training images
    :param y_labelled: Labels for labelled training set
    :param y_unlabelled: Labels for 'unlabelled' training set
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

    dataset = tf.data.Dataset.from_tensor_slices((x_labelled, x_unlabelled, y_labelled, y_unlabelled))
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

def test_input_fn(x_test_labelled, y_test_labelled, x_test_unlabelled, y_test_unlabelled, batch_size, buffer_size):
    '''
    Function for creating the input iterator for testing the model
    :param x_test_labelled: Labelled test images
    :param x_test_unlabelled: Unlabelled test images
    :param y_test_labelled: Labels for labelled test images
    :param y_test_unlabelled: Labels for 'unlabelled' test images
    :param batch_size: Batch size to use for the iterator
    :param buffer_size: Buffer size to use when shuffling the data
    :return:
    '''
    dataset = tf.data.Dataset.from_tensor_slices((x_test_labelled, x_test_unlabelled,
                                                  y_test_labelled, y_test_unlabelled))
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