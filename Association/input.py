import tensorflow as tf

def train_input_fn(x_labelled, x_unlabelled, labels, batch_size, buffer_size, epochs):
    dataset = tf.data.Dataset.from_tensor_slices((x_labelled, x_unlabelled, labels))
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, epochs))
    dataset = dataset.batch(batch_size).prefetch(1)
    iter = dataset.make_initializable_iterator()
    xlb, xub, yb = iter.get_next()
    iter_init = iter.initializer

    return {
        'iter_init': iter_init,
        'x_labelled': xlb,
        'x_unlabelled': xub,
        'labels': yb
    }


def test_input_fn(x_test_labelled, x_test_unlabelled, labelled_labels, unlabelled_labels, buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices((x_test_labelled, x_test_unlabelled, labelled_labels, unlabelled_labels))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.prefetch(1)
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