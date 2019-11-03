import os
import numpy as np
import tensorflow as tf
from math import floor
import SimpleITK as sitk


def weight_variable(shape, stddev=0.01, trainable=True, name=None):
    # initial = tf.truncated_normal(shape, stddev=stddev)
    # return tf.Variable(initial, trainable=trainable, name=name)
    with tf.variable_scope("kernel", reuse = tf.AUTO_REUSE):
        kernel = tf.get_variable(name = name, shape = shape, trainable = trainable, initializer = tf.truncated_normal_initializer(stddev = stddev))
    return kernel


def residual_block_leaky(x, w1, w2, keep_prob_, is_train, bn_trainable, inc_dim = False, scope = None):
    """
    :param x:
    :param w1:
    :param w2:
    :param keep_prob_:
    :param is_train: set for BN layer, whether in training mode
    :param bn_trainable:
    :param inc_dim:
    :param scope:
    :return:
    """

    _x_channel = x.get_shape().as_list()[-1]
    if scope is None:
        _loc_scope1 = None
        _loc_scope2 = None
    else:
        _loc_scope1 = scope + "_1"
        _loc_scope2 = scope + "_2"

    _inner_conv = bn_leaky_relu_conv2d_layer(x, w1, keep_prob_, is_train = is_train, scope = _loc_scope1, bn_trainable = bn_trainable)
    _inner_conv = bn_leaky_relu_conv2d_layer(_inner_conv, w2, keep_prob_, is_train = is_train, scope = _loc_scope2, bn_trainable = bn_trainable)

    if inc_dim is True:
        # pooled_out = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        x_s = tf.pad(x, [ [0,0], [0,0], [0,0], [_x_channel // 2, _x_channel // 2]])
    else:
        x_s = x

    return x_s + _inner_conv


def bn_leaky_relu_conv2d_layer(x, w, keep_prob_, is_train, scope, bn_trainable, stride = 1):

    bn_layer = batch_norm(x, is_training = is_train, scope = scope, trainable = bn_trainable)
    leaky_relu_layer = tf.nn.leaky_relu(bn_layer)
    conv2d_layer = tf.nn.conv2d(leaky_relu_layer, w, strides=[1,stride,stride,1], padding='SAME')
    return tf.nn.dropout(conv2d_layer, keep_prob_)


def batch_norm(x, is_training, scope, trainable = True):
    # Important: set updates_collections=None to force the updates in place, but that can have a speed penalty, especially in distributed settings.
    # with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
    return tf.contrib.layers.batch_norm(x, is_training = is_training, decay = 0.90, scale = True, center = True, \
                                        scope = scope, variables_collections = ["internal_batchnorm_variables"], \
                                        updates_collections = None, trainable = trainable)


def DR_block_leaky(x, w1, w2, keep_prob_, is_train, rate, bn_trainable, inc_dim = False, scope = None):

    _x_channel = x.get_shape().as_list()[-1]
    if scope is None:
        _loc_scope1 = None
        _loc_scope2 = None
    else:
        _loc_scope1 = scope + "_1"
        _loc_scope2 = scope + "_2"

    _inner_conv = bn_leaky_relu_dilate_conv2d_layer(x, w1, keep_prob_, is_train = is_train, rate = rate, scope = _loc_scope1, bn_trainable = bn_trainable)
    _inner_conv = bn_leaky_relu_dilate_conv2d_layer(_inner_conv, w2, keep_prob_, is_train = is_train, rate =rate, scope = _loc_scope2, bn_trainable = bn_trainable)

    if inc_dim is True:
        # pooled_out = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        x_s = tf.pad(x, [ [0,0], [0,0], [0,0], [_x_channel // 2, _x_channel // 2]])
    else:
        x_s = x

    return x_s + _inner_conv


def bn_leaky_relu_dilate_conv2d_layer(x, w, keep_prob_, is_train, rate, scope, bn_trainable, stride = 1):

    bn_layer = batch_norm(x, is_training = is_train, scope = scope, trainable = bn_trainable)
    leaky_relu_layer = tf.nn.leaky_relu(bn_layer)
    dilate_conv2d = tf.nn.atrous_conv2d(leaky_relu_layer, w, rate = rate, padding = "SAME")
    return tf.nn.dropout(dilate_conv2d, keep_prob_)


def conv2d(x, w):
    conv_2d = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
    return conv_2d


def conv2d_sym(x, w, is_train=True, scope=None, bn_trainable=True):
    # this is for convolution with symmetric padding, to deal with boundary effect!
    # also include bn and relu
    k_shape = w.get_shape().as_list()
    pd_offset = tf.constant( [[0, 0], [  floor(k_shape[0] / 2 ) , floor(k_shape[0] / 2 )], [  floor(k_shape[1] / 2 ) , floor(k_shape[1] / 2)], [0, 0 ]] )
    pd_offset = tf.cast(pd_offset, tf.int32)
    x = tf.pad(x, pd_offset, 'SYMMETRIC' )
    conv_2d = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding = 'VALID')
    bn = batch_norm(conv_2d, is_training = is_train, scope = scope, trainable = bn_trainable)
    return tf.nn.leaky_relu(bn)


def conv2d_sym_only(x, w, stride=1):
    # this is for convolution with symmetric padding, to deal with boundary effect!
    k_shape = w.get_shape().as_list()
    pd_offset = tf.constant( [[0, 0], [  floor(k_shape[0] / 2 ) , floor(k_shape[0] / 2 )], [  floor(k_shape[1] / 2 ) , floor(k_shape[1] / 2)], [0, 0 ]] )
    pd_offset = tf.cast(pd_offset, tf.int32)
    x = tf.pad(x, pd_offset, 'SYMMETRIC' )
    conv_2d = tf.nn.conv2d(x, w, strides=[1,stride,stride,1], padding = 'VALID')
    return conv_2d


def max_pool2d(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')


def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1,  tf.shape(output_map)[3]]))
    return tf.clip_by_value( tf.div(exponential_map,tensor_sum_exp), -1.0 * 1e15, 1.0* 1e15, name = "pixel_softmax_2d")


def _read_lists(fid):
    """ read train list and test list """
    if not os.path.isfile(fid):
        return None
    with open(fid, 'r') as fd:
        _list = fd.readlines()

    my_list = []
    for _item in _list:
        if len(_item) < 5:
            _list.remove(_item)
        my_list.append(_item.split('\n')[0])
    return my_list


def _label_decomp(label_vol, num_class):
    """decompose label for softmax classifier
        original labels are batchsize * W * H * 1, with label values 0,1,2,3...
        this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
        numpy version of tf.one_hot
    """
    _batch_shape = list(label_vol.shape)
    _vol = np.zeros(_batch_shape)
    _vol[label_vol == 0] = 1
    _vol = _vol[..., np.newaxis]
    for i in range(num_class):
        if i == 0:
            continue
        _n_slice = np.zeros(label_vol.shape)
        _n_slice[label_vol == i] = 1
        _vol = np.concatenate((_vol, _n_slice[..., np.newaxis]), axis=3)
    return np.float32(_vol)


def _phase_shift(I, r, batch_size):
    # Helper function with main phase shift operation

    _, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (batch_size, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
    if batch_size == 1:
        X = tf.expand_dims( X, 0 )
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    if batch_size == 1:
        X = tf.concat([x for x in X], 2 )
    else:
        X = tf.concat([tf.squeeze(x) for x in X], 2)  #
    out =  tf.reshape(X, (batch_size, a*r, b*r, 1))
    if batch_size == 1:
        out = tf.transpose( out, (0,2,1,3)  )
    return out


def PS(X, r, batch_size, n_channel = 8):
  # Main OP that you can arbitrarily use in you tensorflow code
    Xc = tf.split(X, n_channel, -1 )
    X = tf.concat([_phase_shift(x, r, batch_size) for x in Xc], 3)
    return X


def _eval_dice(gt_y, pred_y, detail=False):

    class_map = {  # a map used for mapping label value to its name, used for output
        "0": "bg",
        "1": "lv_myo",
        "2": "la_blood",
        "3": "lv_blood",
        "4": "aa"
    }

    dice = []

    for cls in xrange(1, 5):

        gt = np.zeros(gt_y.shape)
        pred = np.zeros(pred_y.shape)

        gt[gt_y == cls] = 1
        pred[pred_y == cls] = 1

        dice_this = 2*np.sum(gt*pred)/(np.sum(gt)+np.sum(pred))
        dice.append(dice_this)

        if detail is True:
            print ("class {}, dice is {:2f}".format(class_map[str(cls)], dice_this))

    return dice


def _save_nii(pred_mask, gt_mask, gt_fle, output_path):

    ref = sitk.ReadImage(gt_fle)

    img = sitk.GetImageFromArray(pred_mask)
    img.SetSpacing(ref.GetSpacing())
    img.SetOrigin(ref.GetOrigin())
    img.SetDirection(ref.GetDirection())

    save_path = os.path.join(output_path, gt_fle.split('/')[-1].split('.')[0]+'_predmask.nii.gz')
    sitk.WriteImage(img, save_path)

    gt_mask[gt_mask > 4] = 0
    img = sitk.GetImageFromArray(gt_mask)
    img.SetSpacing(ref.GetSpacing())
    img.SetOrigin(ref.GetOrigin())
    img.SetDirection(ref.GetDirection())

    save_path = os.path.join(output_path, gt_fle.split('/')[-1].split('.')[0]+'_gtmask.nii.gz')
    sitk.WriteImage(img, save_path)

