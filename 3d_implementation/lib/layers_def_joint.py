import tensorflow as tf
import tensorflow.contrib.slim as slim

''' Fundamental 3D Convolution Definition '''


# 3D convolution
def conv3d(inputs, output_channels, kernel_size, stride, padding='same', use_bias=False, name='conv'):
    tensor = tf.layers.conv3d(
        inputs=inputs,                  # Tensor input
        filters=output_channels,        # Integer, the dimensionality of the output space
        kernel_size=kernel_size,        # An integer or tuple/list of 3, depth, height and width
        strides=stride,                 # (1, 1, 1)
        padding=padding,                # "valid" or "same", same: zero padding
        data_format='channels_last',    # channels_last (batch, depth, height, width, channels)
                                        # channels_first (batch, channels, depth, height, width)
        dilation_rate=(1, 1, 1),        # incompatible problem with stride value != 1
        activation=None,                # None to maintain a linear activation
        use_bias=use_bias,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        # kernel_regularizer=slim.l2_regularizer(scale=0.0005),
        bias_initializer=tf.zeros_initializer(),
        name=name,                      # the name of the layer
        # reuse=None                    # reuse the weights of a previous layer by the same name
        # bias_regularizer = None,
        # activity_regularizer = None,  # Regularizer function for the output
        # trainable = True,
    )
    return tensor

def batch_norm(x, is_training, scope = None, trainable = True):
    # Important: set updates_collections=None to force the updates in place, but that can have a speed penalty, especially in distributed settings.
    return tf.contrib.layers.batch_norm(x, is_training = is_training, decay = 0.50, scale = True, center = True, \
                                        scope = scope+'/batch_norm', variables_collections = ["internal_batchnorm_variables"], \
                                        updates_collections = None, trainable = trainable)

#Convolution, Batch normalization, ReLU unit
# def conv_bn_relu(inputs, output_channels, kernel_size, stride, is_training, keep_prob, name, padding='same', use_bias=False):
#     with tf.variable_scope(name):
#         conv = conv3d(inputs, output_channels, kernel_size, stride, padding=padding, use_bias=use_bias, name=name+'_conv')
#         bn = tf.layers.batch_normalization(
#             inputs=conv,                # tensor, first dimension of batch_size
#             center=True,
#             scale=True,                 # If True, multiply by gamma. If False, gamma is not used
#             # epsilon=1e-5,               # Small float added to variance to avoid dividing by zero
#             # updates_collections=None,   # tf.GraphKeys.UPDATE_OPS,
#             # updates_collections: Collections to collect the update ops for computation.
#             # The updates_ops need to be executed with the train_op.
#             # If None, a control dependency would be added to make sure the updates are computed in place
#             training=is_training,
#             # In training mode it would accumulate the statistics of the moments into moving_mean
#             # and moving_variance using an exponential moving average with the given decay.
#             #name=name+'_batch_norm',         # variable_scope
#             # reuse=None,
#             #variables_collections=["internal_batchnorm_variables"],
#             # outputs_collections=None,
#             # trainable=True,
#             # batch_weights=None,
#             # fused=False,
#             # data_format=DATA_FORMAT_NHWC,
#             # zero_debias_moving_mean=False,
#             # renorm=False,
#             # renorm_clipping=None,
#             # renorm_decay=0.99,
#             # param_initializers = None,
#             # param_regularizers = None,
#             # activation_fn = None,
#         )

#         relu = tf.nn.leaky_relu(features=bn, name=name+'_relu')
#         # dropped = tf.nn.dropout(relu, keep_prob)

#     return relu
def conv_bn_relu(inputs, output_channels, kernel_size, stride, is_training, keep_prob, name, padding='same', use_bias=False):
    with tf.variable_scope(name):
        conv = conv3d(inputs, output_channels, kernel_size, stride, padding=padding, use_bias=use_bias, name=name+'_conv')
        bn = batch_norm(conv, is_training = is_training, scope = name, trainable =True)
        relu = tf.nn.leaky_relu(features=bn, name=name+'_relu')
        # dropped = tf.nn.dropout(relu, keep_prob)

    return relu

def conv_in_relu (inputs, output_channels, kernel_size, stride, keep_prob, name, padding='same', use_bias=False):
    conv = conv3d(inputs, output_channels, kernel_size, stride, padding=padding, use_bias=use_bias, name=name + '_conv')
    ins = tf.contrib.layers.instance_norm(inputs=conv, center=True, scale=True, trainable=True)  # data_format=DATA_FORMAT_NHWC as default
    relu = tf.nn.leaky_relu(features=ins, name=name+'_relu')

    return relu


# 3D Deconvolution
def deconv3d(inputs, output_channels, name='deconv'):
    # depth, height and width
    batch, in_depth, in_height, in_width, in_channels = [int(d) for d in inputs.get_shape()]
    dev_filter = tf.get_variable(
        name=name+'/filter',          # name of the new or existing variable
        # shape=[4, 4, 4, output_channels, in_channels],
        shape=[3, 3, 3, output_channels, in_channels],
        # 4, 4, 4, depth, height and width
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
        regularizer=slim.l2_regularizer(scale=0.0005),
        # trainable=True,
        # collections=None,
        # caching_device=None,
        # partitioner=None,
        # validate_shape=True,
        # use_resource=None,
        # custom_getter=None
    )
    deconv = tf.nn.conv3d_transpose(
        value=inputs,                   # [batch, depth, height, width, in_channels]
        filter=dev_filter,              # [depth, height, width, output_channels, in_channels]
        output_shape=[batch, in_depth*2, in_height*2, in_width*2, output_channels],
        strides=[1, 2, 2, 2, 1],
        padding='SAME',
        data_format='NDHWC',
        name=name
    )
    '''Strides and Filter shape, draw the pictures'''
    return deconv


#3D Deconvolution, Batch normalization, ReLU unit
# def deconv_bn_relu(inputs, output_channels, is_training, keep_prob, name):
#     with tf.variable_scope(name):
#         deconv = deconv3d(inputs=inputs, output_channels=output_channels, name=name+'_deconv')
#         bn = tf.layers.batch_normalization(inputs=deconv, center=True, scale=True, training=is_training)
#         relu = tf.nn.leaky_relu(features=bn, name=name+'_lrelu')
#         # dropped = tf.nn.dropout(relu, keep_prob)
#     return relu
def deconv_bn_relu(inputs, output_channels, is_training, keep_prob, name):
    with tf.variable_scope(name):
        deconv = deconv3d(inputs=inputs, output_channels=output_channels, name=name+'_deconv')
        bn = batch_norm(deconv, is_training=is_training, scope=name)
        relu = tf.nn.leaky_relu(features=bn, name=name+'_lrelu')
        # dropped = tf.nn.dropout(relu, keep_prob)
    return relu

def deconv_in_relu(inputs, output_channels, keep_prob, name):
    with tf.variable_scope(name):
        deconv = deconv3d(inputs=inputs, output_channels=output_channels, name=name+'_deconv')
        ins = tf.contrib.layers.instance_norm(inputs=deconv, center=True, scale=True, trainable=True)  # data_format=DATA_FORMAT_NHWC as default
        relu = tf.nn.leaky_relu(features=ins, name=name+'_lrelu')
        # dropped = tf.nn.dropout(relu, keep_prob)
    return relu


# # 3 Units of conv_bn_relu with combined output, why?
# def conv_bn_relu_x3(inputs, output_channels, kernel_size, stride, is_training, name,
#                     padding='same', use_bias=False):
#     with tf.variable_scope(name):
#         z = conv_bn_relu(inputs, output_channels, kernel_size, stride, is_training, name=name+'_dense1',
#                          padding=padding, use_bias=use_bias)
#         z_out = conv_bn_relu(z, output_channels, kernel_size, stride, is_training, name=name+'_dense2',
#                              padding=padding, use_bias=use_bias)
#         z_out = conv_bn_relu(z_out, output_channels, kernel_size, stride, is_training, name=name+'_dense3',
#                              padding=padding, use_bias=use_bias)
#     return z+z_out
#     # input -> z -> z_out -> z_out + z


def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 4, keepdims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, 1, tf.shape(output_map)[4]]))
    return tf.clip_by_value( tf.div(exponential_map,tensor_sum_exp), -1.0 * 1e15, 1.0* 1e15, name = "pixel_softmax_2d")