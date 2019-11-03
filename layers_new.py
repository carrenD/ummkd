

from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow as tf
from math import floor


def weight_variable(shape, stddev=0.01, trainable=True, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, trainable=trainable, name=name)


def sharable_weight_variable(shape, stddev=0.01, trainable = True, name = None):
    """
    sharable through variable scope reuse
    """
    return tf.get_variable(name = name, shape = shape, trainable = trainable, initializer = tf.truncated_normal_initializer(stddev = stddev))


def max_pool2d(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def conv2d(x, w):
    """
    :param x: input to the layer
    :param W: convolution kernel
    :param keep_prob_: keep rate for dropout
    :return: convolution results with dropout
    """
    conv_2d = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
    return conv_2d

def conv2d_sym_only(x, w, stride=1):
    # this is for convolution with symmetric padding, to deal with boundary effect!
    k_shape = w.get_shape().as_list()
    pd_offset = tf.constant( [[0, 0], [  floor(k_shape[0] / 2 ) , floor(k_shape[0] / 2 )], [  floor(k_shape[1] / 2 ) , floor(k_shape[1] / 2)], [0, 0 ]] )
    pd_offset = tf.cast(pd_offset, tf.int32)
    x = tf.pad(x, pd_offset, 'SYMMETRIC' )
    return tf.nn.conv2d(x, w, strides=[1,stride,stride,1], padding = 'VALID')

def conv2d_sym(x, w, is_train=True, scope=None, bn_trainable=True):
    # this is for convolution with symmetric padding, to deal with boundary effect!
    # also include bn and relu
    k_shape = w.get_shape().as_list()
    pd_offset = tf.constant( [[0, 0], [  floor(k_shape[0] / 2 ) , floor(k_shape[0] / 2 )], [  floor(k_shape[1] / 2 ) , floor(k_shape[1] / 2)], [0, 0 ]] )
    pd_offset = tf.cast(pd_offset, tf.int32)
    x = tf.pad(x, pd_offset, 'SYMMETRIC' )
    conv_2d = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding = 'VALID')
    bn = batch_norm(conv_2d, is_training = is_train, scope = scope, trainable = bn_trainable)
    return tf.nn.relu(bn)


def bn_leaky_relu_conv2d_sym_layer(x, w, keep_prob_, stride=1, is_train=True, scope=None, bn_trainable=True):

    bn_layer = batch_norm(x, is_training = is_train, scope = scope, trainable = bn_trainable)
    relu_layer = tf.nn.leaky_relu(bn_layer)
    conv2d_layer = conv2d_sym_only(relu_layer, w, stride=stride)
    return tf.nn.dropout(conv2d_layer, keep_prob_)


def bn_relu_conv2d_layer(x, w, keep_prob_, stride=1, is_train=True, scope=None, bn_trainable=True):

    bn_layer = batch_norm(x, is_training = is_train, scope = scope, trainable = bn_trainable)
    relu_layer = tf.nn.relu(bn_layer)
    conv2d_layer = tf.nn.conv2d(relu_layer, w, strides=[1,stride,stride,1], padding='SAME')

    return tf.nn.dropout(conv2d_layer, keep_prob_)


def bn_leaky_relu_conv2d_layer(x, w, keep_prob_, stride=1, is_train=True, scope=None, bn_trainable=True):

    bn_layer = batch_norm(x, is_training = is_train, scope = scope, trainable = bn_trainable)
    relu_layer = tf.nn.leaky_relu(bn_layer)
    conv2d_layer = tf.nn.conv2d(relu_layer, w, strides=[1,stride,stride,1], padding='SAME')

    return tf.nn.dropout(conv2d_layer, keep_prob_)


def dilate_conv2d(x, w, rate = 2):

    di_conv_2d = tf.nn.atrous_conv2d(x, w, rate = rate, padding = 'SAME')

    # elif padding == 'SYMMETRIC':
    #     k_shape = W.get_shape().as_list()
    #     pd_offset = tf.constant( [[0, 0], [  floor(k_shape[0] / 2 ) , floor(k_shape[0] / 2 )], [  floor(k_shape[1] / 2 ) , floor(k_shape[1] / 2)], [0, 0 ]] )
    #     pd_offset = tf.cast(pd_offset, tf.int32)
    #     x = tf.pad(x, pd_offset, 'SYMMETRIC' )
    #     di_conv_2d = tf.nn.atrous_conv2d(x, W, rate = rate, padding = 'VALID')
    # return tf.nn.dropout(di_conv_2d, keep_prob_)

    return di_conv_2d


def bn_relu_dilate_conv2d_layer(x, w, keep_prob_, is_train=True, scope=None, bn_trainable=True):

    bn_layer = batch_norm(x, is_training = is_train, scope = scope, trainable = bn_trainable)
    relu_layer = tf.nn.relu(bn_layer)
    conv2d_layer = dilate_conv2d(relu_layer, w)

    return tf.nn.dropout(conv2d_layer, keep_prob_)


def bn_leaky_relu_dilate_conv2d_layer(x, w, keep_prob_, is_train=True, scope=None, bn_trainable=True):

    bn_layer = batch_norm(x, is_training = is_train, scope = scope, trainable = bn_trainable)
    relu_layer = tf.nn.leaky_relu(bn_layer)
    conv2d_layer = dilate_conv2d(relu_layer, w)

    return tf.nn.dropout(conv2d_layer, keep_prob_)


def batch_norm_indeed(x, is_training, scope = None, trainable = True):
    # Important: set updates_collections=None to force the updates in place, but that can have a speed penalty, especially in distributed settings.
    return tf.contrib.layers.batch_norm(x, is_training = is_training, decay = 0.90, scale = True, center = True, \
                                        scope = scope, variables_collections = ["internal_batchnorm_variables"], \
                                        updates_collections = None, trainable = trainable)


def batch_norm(x, is_training, scope = None, trainable = True):
    # Here is actually using group normalization, but we give its name as batch_norm as well, to reuse all other implementations as a quick testing idea
    print ('using group norm ...')
    return tf.contrib.layers.group_norm(inputs = x, groups = 16, center = True, scale = True, scope = scope, trainable = trainable)



def residual_block_leaky(x, w1, w2, keep_prob_, stride=1, inc_dim = False, is_train = True, scope = None, bn_trainable = True, leak = False):
    """Args:
        adapt_scope: a flag indicating the variable scope for batch_norm
        what else can i do? tensorflow sucks!
    param: scope: setting for batch_norm variables
    """
    _x_channel = x.get_shape().as_list()[-1]
    if scope is None:
        _loc_scope1 = None
        _loc_scope2 = None
    else:
        _loc_scope1 = scope + "_1"
        _loc_scope2 = scope + "_2"

    # if inc_dim is True:
    #     stride = 2

    _inner_conv = bn_leaky_relu_conv2d_layer(x, w1, keep_prob_, is_train=is_train, scope=_loc_scope1, bn_trainable=bn_trainable)
    _inner_conv = bn_leaky_relu_conv2d_layer(_inner_conv, w2, keep_prob_, is_train=is_train, scope=_loc_scope2, bn_trainable=bn_trainable)

    if inc_dim is True:
        # pooled_out = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        x_s = tf.pad(x, [ [0,0], [0,0], [0,0], [_x_channel // 2, _x_channel // 2]])
    else:
        x_s = x

    return x_s + _inner_conv


def residual_block(x, w1, w2, keep_prob_, stride=1, inc_dim = False, is_train = True, scope = None, bn_trainable = True, leak = False):
    """
        Args:
        is_train: whether the mode is training, for setting of the bn layer, moving_mean and moving_variance
        param: scope: setting for batch_norm variables
    """
    _x_channel = x.get_shape().as_list()[-1]
    if scope is None:
        _loc_scope1 = None
        _loc_scope2 = None
    else:
        _loc_scope1 = scope + "_1"
        _loc_scope2 = scope + "_2"

    # if inc_dim is True:
    #     stride = 2

    _inner_conv = bn_relu_conv2d_layer(x, w1, keep_prob_, is_train=is_train, scope=_loc_scope1, bn_trainable=bn_trainable)
    _inner_conv = bn_relu_conv2d_layer(_inner_conv, w2, keep_prob_, is_train=is_train, scope=_loc_scope2, bn_trainable=bn_trainable)

    if inc_dim is True:
        # pooled_out = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        x_s = tf.pad(x, [ [0,0], [0,0], [0,0], [_x_channel // 2, _x_channel // 2]])
    else:
        x_s = x

    return x_s + _inner_conv


## dilated conv for con bn relu block
def DR_block(x, w1, w2, keep_prob, rate, inc_dim = False, is_train = True, bn_trainable = True, scope = None, leak = False):
    _x_channel = x.get_shape().as_list()[-1]

    if scope is None:
        _loc_scope1 = None
        _loc_scope2 = None
    else:
        _loc_scope1 = scope + "_1"
        _loc_scope2 = scope + "_2"

    _inner_conv = bn_relu_dilate_conv2d_layer(x, w1, keep_prob_ = keep_prob, is_train = is_train, scope = _loc_scope1, bn_trainable = bn_trainable)
    _inner_conv = bn_relu_dilate_conv2d_layer(_inner_conv, w2, keep_prob_ = keep_prob, is_train = is_train, scope = _loc_scope2, bn_trainable = bn_trainable)

    if inc_dim is True:
        x_s = tf.pad(x, [ [0,0], [0,0], [0,0], [_x_channel // 2, _x_channel // 2]])
    else:
        x_s = x

    return x_s + _inner_conv



## dilated conv for con bn relu block
def DR_block_leaky(x, w1, w2, keep_prob, rate, inc_dim = False, is_train = True, bn_trainable = True, scope = None, leak = False):
    _x_channel = x.get_shape().as_list()[-1]

    if scope is None:
        _loc_scope1 = None
        _loc_scope2 = None
    else:
        _loc_scope1 = scope + "_1"
        _loc_scope2 = scope + "_2"

    _inner_conv = bn_leaky_relu_dilate_conv2d_layer(x, w1, keep_prob_ = keep_prob, is_train = is_train, scope = _loc_scope1, bn_trainable = bn_trainable)
    _inner_conv = bn_leaky_relu_dilate_conv2d_layer(_inner_conv, w2, keep_prob_ = keep_prob, is_train = is_train, scope = _loc_scope2, bn_trainable = bn_trainable)

    if inc_dim is True:
        x_s = tf.pad(x, [ [0,0], [0,0], [0,0], [_x_channel // 2, _x_channel // 2]])
    else:
        x_s = x

    return x_s + _inner_conv



def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1,  tf.shape(output_map)[3]]))
    return tf.clip_by_value( tf.div(exponential_map,tensor_sum_exp), -1.0 * 1e15, 1.0* 1e15, name = "pixel_softmax_2d")


def simple_concat2d(x1,x2):
    """ concatenation without offset check"""
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    try:
        tf.equal(x1_shape[0:-2], x2_shape[0: -2])
    except:
        print("x1_shape: %s"%str(x1.get_shape().as_list()))
        print("x2_shape: %s"%str(x2.get_shape().as_list()))
        raise ValueError("Cannot concatenate tensors with different shape, igonoring feature map depth")
    return tf.concat([x1, x2], 3)
