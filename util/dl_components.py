import tensorflow as tf


def get_weights(name, shape, initializer):
    return tf.get_variable(name, dtype=tf.float32, shape=shape, initializer=initializer)


def _modified_smooth_l1(sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    """
    sigma2 = sigma * sigma

    inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                              tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

    outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

    return outside_mul


def do_reshape(input, d, name):
    input_shape = tf.shape(input)
    if name == 'rpn_cls_prob_reshape':
        # the shape of input is N, W/2*18, H, C
        # the shape of mid is N, 18, W, H
        # the shape of output is N, W, H, 18
        return tf.transpose(tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [input_shape[0],
                                                                           int(d), tf.cast(
                tf.cast(input_shape[1], tf.float32) / tf.cast(d, tf.float32) * tf.cast(input_shape[3], tf.float32),
                tf.int32), input_shape[2]]), [0, 2, 3, 1], name=name)
    else:
        return tf.transpose(tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [input_shape[0],
                                                                           int(d), tf.cast(
                tf.cast(input_shape[1], tf.float32) * (
                    tf.cast(input_shape[3], tf.float32) / tf.cast(d, tf.float32)), tf.int32), input_shape[2]]),
                            [0, 2, 3, 1], name=name)


def do_conv(x, layer_name, kernel_size, filter_size, stride_size, padding='SAME', activation_method=None):
    with tf.variable_scope(layer_name):
        in_shape = x.get_shape().as_list()
        weights = get_weights('weights', shape=[kernel_size[0], kernel_size[1], in_shape[-1], filter_size],
                              initializer=tf.contrib.layers.xavier_initializer())
        bias = get_weights('bias', shape=[filter_size], initializer=tf.constant_initializer(value=0.0))
        output = tf.nn.conv2d(x, filter=weights,
                              strides=[1, stride_size[0], stride_size[1], 1], padding=padding)
        output = tf.nn.bias_add(output, bias)
        if activation_method is not None:
            output = activation_method(output)
        return output


def do_maxpooling(x, layer_name, kernel_size, stride_size, padding='SAME'):
    with tf.variable_scope(layer_name):
        output = tf.nn.max_pool(x, ksize=[1, kernel_size[0], kernel_size[1], 1],
                                strides=[1, stride_size[0], stride_size[1], 1], padding=padding)
    return output


def do_upconv(x, layer_name, kernel_size, filter_size, output_shape, stride_size, padding, activation_method=None):
    with tf.variable_scope(layer_name):
        in_shape = x.get_shape().as_list()
        weights = get_weights('weights', shape=[kernel_size[0], kernel_size[1], filter_size, in_shape[-1]],
                              initializer=tf.contrib.layers.xavier_initializer())
        output = tf.nn.conv2d_transpose(x, weights, output_shape=output_shape,
                                        strides=[1, stride_size[0], stride_size[1], 1],
                                        padding=padding)
        bias = get_weights('bias', shape=[filter_size], initializer=tf.constant_initializer(value=0.0))
        output = tf.nn.bias_add(output, bias)
        if activation_method is not None:
            output = activation_method(output)
    return output


def do_fc(x, layer_name, output_node, relu=True, trainable=True):
    with tf.variable_scope(layer_name):
        if isinstance(x, tuple):
            x = x[0]
        input_shape = x.get_shape()
        if input_shape.ndims == 4:
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            feed_in = tf.reshape(x, [-1, dim])
        else:
            feed_in, dim = (x, int(input_shape[-1]))
        if layer_name == 'bbox_pred':
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_biases = tf.constant_initializer(0.0)
        else:
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
        weights = tf.get_variable('weights', [dim, output_node], initializer=init_weights, trainable=trainable)
        bias = tf.get_variable('bias', [output_node], initializer=init_biases, trainable=trainable)

        op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
        fc = op(feed_in, weights, bias, name=layer_name)
        return fc