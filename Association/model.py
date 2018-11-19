import tensorflow as tf

def compute_logits(data_in, num_classes, reuse):
    logs = tf.layers.dense(inputs=data_in, units=num_classes, reuse=reuse)
    #For single evals
    if logs.shape[0] == 1:
        logs = tf.reshape(logs, [num_classes])
    return logs

def classification_loss(logits,
                        labels,
                        num_classes):
    with tf.variable_scope('classification'):
        one_hot = tf.one_hot(labels, depth=num_classes)
        #For single evals
        if one_hot.shape[0] == 1:
            one_hot = tf.reshape(one_hot, [num_classes])
        classification_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=one_hot,
            logits=logits
        )
        return classification_loss

def association_loss(labelled_embedding,
                     unlabelled_embedding,
                     labels,
                     batch_size,
                     walker_loss_weight,
                     visit_loss_weight):

    with tf.variable_scope('association'):
        equality_matrix = tf.equal(tf.reshape(labels, [-1, 1]), labels)
        equality_matrix = tf.cast(equality_matrix, tf.float32)
        p_target = (equality_matrix / tf.reduce_sum(
            equality_matrix, [1], keepdims=True))
        match_ab = tf.matmul(labelled_embedding, unlabelled_embedding, transpose_b=True)
        p_ab = tf.nn.softmax(match_ab)
        p_ba = tf.nn.softmax(tf.transpose(match_ab))
        p_aba = tf.matmul(p_ab, p_ba)

        walker_loss = tf.losses.softmax_cross_entropy(
            p_target,
            tf.log(1e-8 + p_aba))
        walker_loss= (walker_loss_weight * walker_loss)

        visit_probability = tf.reduce_mean(p_ab, [0], keepdims=True)
        t_nb = tf.shape(p_ab)[1]
        visit_loss = tf.losses.softmax_cross_entropy(
            tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
            tf.log(1e-8 + visit_probability))
        visit_loss = (visit_loss_weight * visit_loss)

        tf.summary.scalar('walker_loss', walker_loss)
        tf.summary.scalar('visit_loss', visit_loss)

        return walker_loss + visit_loss

def total_loss(labelled_embeddings,
               unlabelled_embeddings,
               labels,
               logits,
               # use_assoc,
               association_loss_weight,
               walker_loss_weight,
               visit_loss_weight,
               batch_size,
               num_classes):
    with tf.variable_scope('total_loss'):
        class_loss = classification_loss(logits, labels, num_classes)
        assoc_loss = association_loss_weight *\
                     association_loss(labelled_embeddings,
                                      unlabelled_embeddings,
                                      labels,
                                      batch_size,
                                      walker_loss_weight,
                                      visit_loss_weight)
                                      # use_assoc)
        l2_loss = tf.losses.get_regularization_loss()
        tf.summary.scalar('l2_loss', l2_loss)
        tf.summary.scalar('class_loss', class_loss)
        tf.summary.scalar('assoc_loss', assoc_loss)
        return class_loss + assoc_loss + l2_loss

def find_embedding(features,
                   dense_dropout_rate,
                   conv_dropout_rate,
                   l2_mag,
                   reuse,
                   filters=(32, 64, 128),
                   conv_kernel_size=3,
                   pool_size=2,
                   conv_stride=1,
                   pool_stride=1,
                   emb_activation='elu',
                   output_units=128):
    with tf.variable_scope('embedding'):
        x = features
        i = 0
        l2 = tf.contrib.layers.l2_regularizer(scale=l2_mag)
        for f in filters:
            i += 1
            with tf.variable_scope('conv_block_{}'.format(i), reuse=reuse):
                x = tf.layers.conv2d(
                    inputs=x,
                    filters=f,
                    kernel_size=conv_kernel_size,
                    strides=conv_stride,
                    activation='elu',
                    kernel_regularizer=l2,
                    name='conv2d_1'
                )
                x = tf.layers.conv2d(
                    inputs=x,
                    filters=f,
                    kernel_size=conv_kernel_size,
                    strides=conv_stride,
                    activation='elu',
                    kernel_regularizer=l2,
                    name='conv2d_2'
                )
                x = tf.layers.dropout(
                    inputs=x,
                    rate=conv_dropout_rate
                )
                x = tf.layers.max_pooling2d(
                    inputs=x,
                    pool_size=pool_size,
                    strides=pool_stride,
                    name='max_pool'
                )
        with tf.variable_scope('fc_emb', reuse=reuse):
            x = tf.layers.flatten(inputs=x, name='flatten')
            x = tf.layers.dense(
                inputs=x,
                units=output_units,
                activation=emb_activation,
                name='dense'
            )
            emb = tf.layers.dropout(
                inputs=x,
                rate=dense_dropout_rate
            )

        return emb

def get_input(inputs,
              predict_mode,
              params):
    l_features = inputs['x_labelled']
    u_features = inputs['x_unlabelled']
    l_features = tf.cast(l_features, tf.float32)
    u_features = tf.cast(u_features, tf.float32)

    # Reshaping MNIST images
    if params['reshape'] == 'l':
        l_features = tf.image.resize_images(l_features, params['target_shape'])
        l_features = tf.image.grayscale_to_rgb(l_features)
    if params['reshape'] == 'u':
        u_features = tf.image.resize_images(u_features, params['target_shape'])
        u_features = tf.image.grayscale_to_rgb(u_features)

    # Eval
    if not predict_mode:
        l_labels = inputs['labels']['labelled']
        u_labels = inputs['labels']['unlabelled']
    # Train / predict
    else:
        l_labels = inputs['labels']
        u_labels = None

    return l_features, u_features, l_labels, u_labels

def get_embedding(u_features,
                  l_features,
                  train_mode,
                  conv_dropout,
                  fc_dropout,
                  params,
                  reuse_first=False):

    u_emb = find_embedding(u_features,
                           fc_dropout,
                           conv_dropout,
                           params['l2_mag'],
                           reuse_first)

    l_emb = find_embedding(l_features,
                           fc_dropout,
                           conv_dropout,
                           params['l2_mag'],
                           True)
    u_emb = tf.cast(u_emb, tf.float32)
    l_emb = tf.cast(l_emb, tf.float32)

    return u_emb, l_emb

def model_fn(inputs,
             mode,
             params):
    eval_mode = mode == tf.estimator.ModeKeys.EVAL
    predict_mode = mode == tf.estimator.ModeKeys.PREDICT
    train_mode = mode == tf.estimator.ModeKeys.TRAIN
    model_spec = inputs

    with tf.variable_scope('input'):
        l_features, u_features, l_labels, u_labels = get_input(inputs, predict_mode, params)
        if params['use_val'] and train_mode:
            vlf, vuf, vll, vul = get_input(inputs['validation'], predict_mode, params)
        else:
            vlf, vuf, vll, vul = None, None, None, None

    association_loss_weight = params['assoc_w']
    walker_loss_weight = params['w_w']
    visit_loss_weight = params['v_w']
    learning_rate = params['lr']
    batch_size = params['batch_size']
    num_classes = params['num_classes']

    with tf.variable_scope('model'):
        u_emb, l_emb = get_embedding(u_features, l_features, train_mode, params['conv_dropout'], params['fc_dropout'], params)
        if params['use_val'] and train_mode:
            vu_emb, vl_emb = get_embedding(vuf, vlf, train_mode, 0.0, 0.0, params, True)
        else:
            vu_emb, vl_emb = None, None

    with tf.variable_scope('logits'):
        if not predict_mode:
            l_logits = compute_logits(l_emb, num_classes, False)
            u_logits = compute_logits(u_emb, num_classes, True)
            if train_mode and params['use_val']:
                vl_logits = compute_logits(vl_emb, num_classes, True)
                vu_logits = compute_logits(vu_emb, num_classes, True)
            else:
                vl_logits, vu_logits = None, None
        else:
            l_logits = compute_logits(l_emb, num_classes, False)
            u_logits, vl_logits, vu_logits = None, None, None


    l_predictions = tf.argmax(l_logits, axis=-1)
    u_predictions = tf.argmax(u_logits, axis=-1) if u_logits is not None else None
    if params['use_val'] and train_mode:
        vl_pred = tf.argmax(vl_logits, axis=1)
        vu_pred = tf.argmax(vu_logits, axis=1)

    with tf.variable_scope('acc'):
        l_accuracy = tf.metrics.accuracy(l_labels, l_predictions)
        u_accuracy = tf.metrics.accuracy(u_labels, u_predictions) if u_predictions is not None else None
        if params['use_val'] and train_mode:
            vl_acc = tf.metrics.accuracy(vll, vl_pred)
            vu_acc = tf.metrics.accuracy(vul, vu_pred)
        else:
            vl_acc, vu_acc = None, None
        if train_mode:
            tf.summary.scalar('labelled_accuracy', l_accuracy[1])
            tf.summary.scalar('unlabelled_accuracy', u_accuracy[1])
            if params['use_val']:
                tf.summary.scalar('validation_labelled_accuracy', vl_acc[1])
                tf.summary.scalar('validation_unlabelled_accuracy', vu_acc[1])

    with tf.variable_scope('loss'):
        if train_mode:
            loss = total_loss(l_emb,
                              u_emb,
                              l_labels,
                              l_logits,
                              # use_assoc,
                              association_loss_weight,
                              walker_loss_weight,
                              visit_loss_weight,
                              batch_size,
                              num_classes)
            tf.summary.scalar('loss', loss)

            if params['use_val']:
                with tf.variable_scope('validation_loss'):
                    v_loss = total_loss(
                        vl_emb,
                        vu_emb,
                        vll,
                        vl_logits,
                        association_loss_weight,
                        walker_loss_weight,
                        visit_loss_weight,
                        batch_size,
                        num_classes
                    )
                    tf.summary.scalar('validation_loss', v_loss)
            else:
                v_loss = None
        else:
            loss, v_loss = None, None

    if train_mode:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.train.get_or_create_global_step() if train_mode else None
        train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    model_spec['var_init'] = tf.global_variables_initializer()
    acc_var = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc")
    model_spec['acc_init'] = tf.variables_initializer(acc_var)
    model_spec['l_prediction'] = l_predictions
    model_spec['l_label'] = l_labels
    model_spec['u_prediction'] = l_predictions
    model_spec['u_label'] = l_labels
    model_spec['loss'] = loss
    model_spec['summaries'] = tf.summary.merge_all()
    model_spec['update_l_acc'] = l_accuracy[1]
    model_spec['l_accuracy'] = l_accuracy[0]
    model_spec['update_u_acc'] = u_accuracy[1] if u_accuracy is not None else None
    model_spec['u_accuracy'] = u_accuracy[0] if u_accuracy is not None else None
    model_spec['val_loss'] = v_loss
    model_spec['val_l_acc'] = vl_acc[1] if vl_acc is not None else None
    model_spec['val_u_acc'] = vu_acc[1] if vu_acc is not None else None
    if train_mode:
        model_spec['train_op'] = train_op
    return model_spec