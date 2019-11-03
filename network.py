import os
import glob
import time
import shutil
import numpy as np
from collections import OrderedDict
import __future__
#from collections import DefaultDict
import logging
# import matplotlib
import tensorflow as tf
# import csv

# import pymedimage.visualize as viz
# import pymedimage.niftiio as nio
from tensorflow.python import debug as tf_debug
from lib.util import *

np.random.seed(0)
contour_map = { # a map used for mapping label value to its name, used for output
    "bg": 0,
    "lv_myo": 1,
    "la_blood": 2,
    "lv_blood": 3,
    "aa": 4
}

logging.basicConfig(filename = "curr_log", level=logging.DEBUG, format='%(asctime)s %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())

raw_size = [256, 256, 3] # original raw input size
volume_size = [256, 256, 3] # volume size after processing, for the tfrecord file
label_size = [256, 256, 1] # size of label


class Build_network(object):

    def __init__ (self, channels, n_class, batch_size, cost_kwargs={}, network_config = {}, opt_kwargs = {}):

        tf.reset_default_graph()

        self.channels = channels
        self.n_class = n_class
        self.batch_size = batch_size

        self.network_config = network_config  # set whether generator/discriminator trainable
        self.opt_kwargs = opt_kwargs

        self.keep_prob = tf.placeholder(dtype=tf.float32, name = "dropout_keep_rate")  # dropout (keep probability

        self.source = tf.placeholder("float", shape=[None, volume_size[0], volume_size[1], self.channels])
        self.source_y = tf.placeholder("float", shape=[None, label_size[0], label_size[1], self.n_class]) # source segmentation
        self.target = tf.placeholder("float", shape=[None, volume_size[0], volume_size[1], self.channels])
        self.target_y = tf.placeholder("float", shape=[None, label_size[0], label_size[1], self.n_class]) # target segmentation
        self.training_mode_source = tf.placeholder_with_default(True, shape = None, name = "training_mode_for_bn_moving_source")
        self.training_mode_target = tf.placeholder_with_default(True, shape = None, name = "training_mode_for_bn_moving_target")

        with tf.variable_scope("segmenter") as scope:
            source_seg_logits = self.segmentation_network(input_data = self.source, keep_prob = self.keep_prob, mode = self.training_mode_source,\
                                                      kernel_collect_trainable = network_config["source_kernel_update"], bn_collect_trainable = network_config["source_bn_update"], bn_scope = "source")

        with tf.variable_scope("segmenter") as scope:
            target_seg_logits = self.segmentation_network(input_data = self.target, keep_prob = self.keep_prob, mode = self.training_mode_target,\
                                                      kernel_collect_trainable = network_config["target_kernel_update"], bn_collect_trainable = network_config["target_bn_update"], bn_scope = "target")


        self.var_list = tf.trainable_variables("segmenter")
        for var in self.var_list:
            print var


        # calculate cost
        self.cost_kwargs = cost_kwargs
        self.source_seg_dice_loss, self.source_seg_ce_loss = self._get_segmentation_cost(seg_logits = source_seg_logits, seg_gt = self.source_y)

        self.target_seg_dice_loss, self.target_seg_ce_loss = self._get_segmentation_cost(seg_logits = target_seg_logits, seg_gt = self.target_y)

        self.kd_loss = self._get_kd_cost(source_logits = source_seg_logits, source_gt = self.source_y, target_logits = target_seg_logits, target_gt = self.target_y)

        self.L2_norm = self._get_L2_Norm(variables_list = self.var_list)

        self.overall_loss = self.source_seg_dice_loss * self.cost_kwargs["miu_seg_dice"] + self.source_seg_ce_loss * self.cost_kwargs["miu_seg_ce"] + \
                            self.target_seg_dice_loss * self.cost_kwargs["miu_seg_dice"] + self.target_seg_ce_loss * self.cost_kwargs["miu_seg_ce"] + \
                            self.kd_loss * self.cost_kwargs["miu_kd"] + self.L2_norm * self.cost_kwargs["miu_seg_L2_norm"]

        # # compute for monitor training procedure
        # # segmentation for source
        self.source_pred_prob = pixel_wise_softmax_2(source_seg_logits)
        self.source_pred_compact = tf.argmax(self.source_pred_prob, 3) # predictions
        self.source_y_compact = tf.argmax(self.source_y, 3) # ground truth
        self.source_dice_eval_arr = self._eval_dice_during_train(self.source_y, self.source_pred_compact)

        # # segmentation for target
        self.target_pred_prob = pixel_wise_softmax_2(target_seg_logits)
        self.target_pred_compact = tf.argmax(self.target_pred_prob, 3) # predictions
        self.target_y_compact = tf.argmax(self.target_y, 3) # ground truth
        self.target_dice_eval_arr = self._eval_dice_during_train(self.target_y, self.target_pred_compact)


    def segmentation_network(self, input_data, mode, keep_prob, kernel_collect_trainable, bn_collect_trainable, bn_scope, feature_base = 16):

        with tf.variable_scope('group_1') as scope:
            w1_1 = weight_variable(shape = [3, 3, self.channels, feature_base], trainable = kernel_collect_trainable, name = "Variable")
            conv1_1 = conv2d(input_data, w1_1)
            wr1_1 = weight_variable(shape = [3, 3, feature_base, feature_base], trainable = kernel_collect_trainable, name = "Variable_1")
            wr1_2 = weight_variable(shape = [3, 3, feature_base, feature_base], trainable = kernel_collect_trainable, name = "Variable_2")
            block1_1 = residual_block_leaky(conv1_1, wr1_1, wr1_2, keep_prob, is_train = mode, bn_trainable = bn_collect_trainable , scope = bn_scope + 'bn_1_1'   ) # here the scope is for bn
            out1 = max_pool2d(block1_1, n = 2)

        with tf.variable_scope('group_2') as scope:
            wr2_1 = weight_variable(shape = [3, 3, feature_base, feature_base * 2], trainable = kernel_collect_trainable, name = "Variable")
            wr2_2 = weight_variable(shape = [3, 3, feature_base * 2, feature_base * 2], trainable = kernel_collect_trainable, name = "Variable_1")
            block2_1 = residual_block_leaky(out1, wr2_1, wr2_2, keep_prob, inc_dim = True,  is_train = mode, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_2_1'  )
            out2 = max_pool2d(block2_1, n = 2)

        with tf.variable_scope('group_3') as scope:
            wr3_1 = weight_variable(shape = [3, 3, feature_base * 2, feature_base * 4], trainable = kernel_collect_trainable, name = "Variable")
            wr3_2 = weight_variable(shape = [3, 3, feature_base * 4, feature_base * 4], trainable = kernel_collect_trainable, name = "Variable_1")
            block3_1 = residual_block_leaky(out2, wr3_1, wr3_2, keep_prob, inc_dim = True, is_train = mode, bn_trainable = bn_collect_trainable , scope = bn_scope + 'bn_3_1')
            wr3_3 = weight_variable(shape = [3, 3, feature_base * 4, feature_base * 4], trainable = kernel_collect_trainable, name = "Variable_2")
            wr3_4 = weight_variable(shape = [3, 3, feature_base * 4, feature_base * 4], trainable = kernel_collect_trainable, name = "Variable_3")
            block3_2 = residual_block_leaky(block3_1, wr3_3, wr3_4, keep_prob, is_train = mode, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_3_2')
            out3 = max_pool2d(block3_2, n = 2)

        with tf.variable_scope('group_4') as scope:
            wr4_1 = weight_variable(shape = [3, 3, feature_base * 4, feature_base * 8], trainable = kernel_collect_trainable, name = "Variable")
            wr4_2 = weight_variable(shape = [3, 3, feature_base * 8, feature_base * 8], trainable = kernel_collect_trainable, name = "Variable_1")
            block4_1 = residual_block_leaky(out3, wr4_1, wr4_2, keep_prob,  inc_dim = True, is_train = mode, bn_trainable = bn_collect_trainable , scope = bn_scope + 'bn_4_1')
            wr4_3 = weight_variable(shape = [3, 3, feature_base * 8, feature_base * 8], trainable = kernel_collect_trainable, name = "Variable_2")
            wr4_4 = weight_variable(shape = [3, 3, feature_base * 8, feature_base * 8], trainable = kernel_collect_trainable, name = "Variable_3")
            block4_2 = residual_block_leaky(block4_1, wr4_3, wr4_4, keep_prob, is_train = mode, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_4_2')

        with tf.variable_scope('group_5') as scope:
            wr5_1 = weight_variable(shape = [3, 3, feature_base * 8, feature_base * 16], trainable = kernel_collect_trainable, name = "Variable")
            wr5_2 = weight_variable(shape = [3, 3, feature_base * 16, feature_base * 16], trainable = kernel_collect_trainable , name = "Variable_1")
            block5_1 = residual_block_leaky(block4_2, wr5_1, wr5_2, keep_prob, inc_dim = True,  is_train = mode, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_5_1')
            wr5_3 = weight_variable(shape = [3, 3, feature_base * 16, feature_base * 16], trainable = kernel_collect_trainable, name = "Variable_2")
            wr5_4 = weight_variable(shape = [3, 3, feature_base * 16, feature_base * 16], trainable = kernel_collect_trainable, name = "Variable_3")
            block5_2 = residual_block_leaky(block5_1, wr5_3, wr5_4, keep_prob, is_train = mode, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_5_2')

        with tf.variable_scope('group_6') as scope:
            wr6_1 = weight_variable(shape = [3, 3, feature_base * 16, feature_base * 16], trainable = kernel_collect_trainable, name = "Variable")
            wr6_2 = weight_variable(shape = [3, 3, feature_base * 16, feature_base * 16], trainable = kernel_collect_trainable, name = "Variable_1")
            block6_1 = residual_block_leaky(block5_2, wr6_1, wr6_2, keep_prob,  is_train = mode, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_6_1')
            wr6_3 = weight_variable(shape = [3, 3, feature_base * 16, feature_base * 16], trainable = kernel_collect_trainable, name = "Variable_2")
            wr6_4 = weight_variable(shape = [3, 3, feature_base * 16, feature_base * 16], trainable = kernel_collect_trainable, name = "Variable_3")
            block6_2 = residual_block_leaky(block6_1, wr6_3, wr6_4, keep_prob, is_train = mode, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_6_2')

        with tf.variable_scope('group_7') as scope: # Since these layers are shared for both MRI/CT, so we use AUTO-REUSE
            wr7_1 = weight_variable(shape = [3, 3, feature_base * 16, feature_base * 32], trainable = kernel_collect_trainable, name = "Variable")
            wr7_2 = weight_variable(shape = [3, 3, feature_base * 32, feature_base * 32], trainable = kernel_collect_trainable, name = "Variable_1")
            block7_1 = residual_block_leaky(block6_2, wr7_1, wr7_2, keep_prob, inc_dim = True,  is_train = mode, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_7_1')
            wr7_3 = weight_variable(shape = [3, 3, feature_base * 32, feature_base * 32], trainable = kernel_collect_trainable, name = "Variable_2")
            wr7_4 = weight_variable(shape = [3, 3, feature_base * 32, feature_base * 32], trainable = kernel_collect_trainable, name = "Variable_3")
            block7_2 = residual_block_leaky(block7_1, wr7_3, wr7_4, keep_prob,  is_train = mode, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_7_2')

        with tf.variable_scope('group_8') as scope:
            wr8_1 = weight_variable(shape = [3, 3, feature_base * 32, feature_base * 32], trainable = kernel_collect_trainable, name = "Variable")
            wr8_2 = weight_variable(shape = [3, 3, feature_base * 32, feature_base * 32], trainable = kernel_collect_trainable, name = "Variable_1")
            block8_1 = DR_block_leaky(block7_2, wr8_1, wr8_2, keep_prob, is_train = mode, rate = 2, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_8_1')
            wr8_3 = weight_variable(shape = [3, 3, feature_base * 32, feature_base * 32], trainable = kernel_collect_trainable, name = "Variable_2")
            wr8_4 = weight_variable(shape = [3, 3, feature_base * 32, feature_base * 32], trainable = kernel_collect_trainable, name = "Variable_3")
            block8_2 = DR_block_leaky(block8_1, wr8_3, wr8_4, keep_prob,  is_train = mode, rate = 2, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_8_2')

        with tf.variable_scope('group_9') as scope:
            w9_1 = weight_variable(shape = [3, 3, feature_base * 32, feature_base * 32], trainable = kernel_collect_trainable, name = "Variable")
            conv9_1 = bn_leaky_relu_conv2d_layer(block8_2, w9_1, keep_prob, is_train = mode, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_9_1')
            w9_2 = weight_variable(shape = [3, 3, feature_base * 32, feature_base * 32], trainable = kernel_collect_trainable, name = "Variable_1")
            conv9_2 = bn_leaky_relu_conv2d_layer(conv9_1, w9_2, keep_prob, is_train = mode, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_9_2')

        with tf.variable_scope('group_10') as scope:
            local_size = 8 * 8 # since the input feature is 8* downsampled, therefore, we need to recover corresponding size. In this case 1 pixel in feature space encodes the laber
            # of a 8*8 region of the original image
            w10_1 = weight_variable( shape = [3, 3, feature_base * 32, local_size * self.n_class * 8], trainable = kernel_collect_trainable, name = "Variable")
            conv10_1 = conv2d_sym(conv9_2, w10_1, is_train = mode, bn_trainable = bn_collect_trainable, scope = bn_scope + 'bn_10_1')
            flat_conv10_1 = PS(conv10_1, r = 8, n_channel = self.n_class * 8, batch_size = self.batch_size) # phase shift

        with tf.variable_scope('output') as scope:
            w11_1 = weight_variable( shape = [5, 5, self.n_class * 8, self.n_class], trainable = kernel_collect_trainable, name = "Variable")
            logits = conv2d_sym_only(flat_conv10_1, w11_1)

        return logits


    def _get_segmentation_cost(self, seg_logits, seg_gt):
        """
        calculate the loss for segmentation prediction
        :param seg_logits: activations before the Softmax function
        :param seg_gt: ground truth segmentaiton mask
        :return: segmentation loss, according to the cost_kwargs setting, cross-entropy weighted loss and dice loss
        """
        softmaxpred = tf.nn.softmax(seg_logits)

        # calculate dice loss, - 2*interesction/union, with relaxed for gradients back-propagation
        dice = 0
        for i in xrange(self.n_class):
            inse = tf.reduce_sum(softmaxpred[:, :, :, i]*seg_gt[:, :, :, i])
            l = tf.reduce_sum(softmaxpred[:, :, :, i]*softmaxpred[:, :, :, i])
            r = tf.reduce_sum(seg_gt[:, :, :, i])
            dice += 2.0 * inse/(l+r+1e-7) # here 1e-7 is relaxation eps
        dice_loss = -1.0 * dice / self.n_class

        # calculate cross-entropy weighted loss
        ce_weighted = 0
        for i in xrange(self.n_class):
            gti = seg_gt[:,:,:,i]
            predi = softmaxpred[:,:,:,i]
            weighted = 1-(tf.reduce_sum(gti) / tf.reduce_sum(seg_gt))
            ce_weighted += -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))
        ce_weighted_loss = tf.reduce_mean(ce_weighted)

        return dice_loss, ce_weighted_loss


    def _get_L2_Norm(self, variables_list):

        L2_norm = sum([tf.nn.l2_loss(var) for var in variables_list if "Variable" in var.name])
        return L2_norm


    def _get_kd_cost(self, source_logits, source_gt, target_logits, target_gt):

        kd_loss = 0.0

        self.source_logits = source_logits
        self.target_logits = target_logits

        self.source_prob = []
        self.target_prob = []

        temperature = 2.0

        for i in xrange(self.n_class):

            eps = 1e-6

            self.s_mask = tf.tile(tf.expand_dims(source_gt[:,:,:,i], -1), [1,1,1,self.n_class])
            self.s_logits_mask_out = self.source_logits * self.s_mask
            self.s_logits_avg = tf.reduce_sum(self.s_logits_mask_out, [0,1,2]) / (tf.reduce_sum(source_gt[:,:,:,i]) + eps)
            self.s_soft_prob = tf.nn.softmax(self.s_logits_avg/temperature)

            self.source_prob.append(self.s_soft_prob)

            self.t_mask = tf.tile(tf.expand_dims(target_gt[:,:,:,i], -1), [1,1,1,self.n_class])
            self.t_logits_mask_out = self.target_logits * self.t_mask
            self.t_logits_avg = tf.reduce_sum(self.t_logits_mask_out, [0,1,2]) / (tf.reduce_sum(target_gt[:,:,:,i]) + eps)
            self.t_soft_prob = tf.nn.softmax(self.t_logits_avg/temperature)

            self.target_prob.append(self.t_soft_prob)

            ## KL divergence loss
            loss = (tf.reduce_sum(self.s_soft_prob * tf.log(self.s_soft_prob/self.t_soft_prob)) + tf.reduce_sum(self.t_soft_prob * tf.log(self.t_soft_prob/self.s_soft_prob))) / 2.0

            ## L2 Norm
            # loss = tf.nn.l2_loss(self.s_soft_prob - self.t_soft_prob) / self.n_class

            kd_loss += loss

        self.kd_loss = kd_loss / self.n_class

        return self.kd_loss


    def _eval_dice_during_train(self, labels, compact_pred):
        """
        calculate standard dice for evaluation, here uses the class prediction, not the probability
        """
        dice_arr = []
        # dice = 0
        eps = 1e-7
        pred = tf.one_hot(compact_pred, depth = self.n_class, axis = -1)
        for i in xrange(self.n_class):
            inse = tf.reduce_sum(pred[:, :, :, i] * labels[:, :, :, i])
            union = tf.reduce_sum(pred[:, :, :, i]) + tf.reduce_sum(labels[:, :, :, i])
            # dice = dice + 2.0 * inse / (union + eps)
            dice_arr.append(2.0 * inse / (union + eps))

        # return 1.0 * dice  / self.n_class, dice_arr

        return dice_arr
