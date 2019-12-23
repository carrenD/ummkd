import os
import glob
import time
import shutil
import numpy as np
from collections import OrderedDict
import __future__
#from collections import DefaultDict
import logging
import tensorflow as tf
from lib.layers_def_bn import *

# np.random.seed(0)

# logging.basicConfig(filename = "curr_log", level=logging.DEBUG, format='%(asctime)s %(message)s')
# logging.getLogger().addHandler(logging.StreamHandler())

class Build_network(object):

    def __init__ (self, batch_size, hyperparam_config={}, cost_kwargs={}):

        logging.info('Using network with reconstruction ...')

        tf.reset_default_graph()

        self.input_channels = hyperparam_config["input_channels"]
        self.patch_size = hyperparam_config["patch_size"] # the dimension is [depth, width, height]
        self.num_class = hyperparam_config["num_class"]
        self.feature_base = hyperparam_config["feature_base"]

        self.batch_size = batch_size

        self.cost_kwargs = cost_kwargs

        self.keep_prob = tf.placeholder(dtype=tf.float32, name = "dropout_keep_rate")  # dropout (keep probability

        self.source = tf.placeholder("float", shape=[self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], self.input_channels])
        self.target = tf.placeholder("float", shape=[self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], self.input_channels])
        self.source_y = tf.placeholder("float", shape=[self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], self.num_class])
        self.target_y = tf.placeholder("float", shape=[self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], self.num_class]) # source segmentation

        self.training_mode = tf.placeholder_with_default(True, shape = None, name = "training_mode_for_bn_moving")

        with tf.variable_scope("joint", reuse = tf.AUTO_REUSE) as scope:
            self.source_feature = self.separate_network(input_data = self.source, is_training = self.training_mode, keep_prob=self.keep_prob, feature_base=self.feature_base, bn_scope='source')
            self.target_feature = self.separate_network(input_data = self.target, is_training = self.training_mode, keep_prob=self.keep_prob, feature_base=self.feature_base, bn_scope='target')

        with tf.variable_scope("joint", reuse = tf.AUTO_REUSE) as scope:
            self.source_seg_logits = self.joint_network(input_data = self.source_feature, is_training = self.training_mode, keep_prob=self.keep_prob, feature_base=self.feature_base, bn_scope='source')
            self.target_seg_logits = self.joint_network(input_data = self.target_feature, is_training = self.training_mode, keep_prob=self.keep_prob, feature_base=self.feature_base, bn_scope='target')

        self.var_list = tf.trainable_variables()
        self.joint_variables = tf.trainable_variables(scope="joint")
        self.global_list = tf.global_variables(scope=None)
        self.source_bn_list = []
        self.target_bn_list = []
        # for var in  self.global_list:
        #      print var
        for var in self.global_list:
            if 'moving_mean' in var.name and 'source' in var.name:
                self.source_bn_list.append(var)
            if 'moving_mean' in var.name and 'target' in var.name:
                self.target_bn_list.append(var)
        for var in self.source_bn_list:
            print var
        for var in self.target_bn_list:
            print var
        # calculate cost
        self.source_seg_dice_loss, self.source_seg_ce_loss, self.source_seg_total_loss = self._get_segmentation_cost(seg_logits = self.source_seg_logits, seg_gt = self.source_y, \
                                                                                                                     variables_list = self.joint_variables, cost_kwargs = self.cost_kwargs)

        self.target_seg_dice_loss, self.target_seg_ce_loss, self.target_seg_total_loss = self._get_segmentation_cost(seg_logits = self.target_seg_logits, seg_gt = self.target_y, \
                                                                                                                     variables_list = self.joint_variables, cost_kwargs = self.cost_kwargs)

        self.weighted_kd_loss = cost_kwargs["miu_kd"] * self._get_kd_cost(source_logits = self.source_seg_logits, source_gt = self.source_y, target_logits = self.target_seg_logits, target_gt = self.target_y)

        self.overall_loss = self.source_seg_total_loss + self.target_seg_total_loss #+ self.weighted_kd_loss
        #self.overall_loss = self.weighted_kd_loss
        # # compute for monitor training procedure
        self.source_pred_prob = pixel_wise_softmax_2(self.source_seg_logits)
        self.source_pred_compact = tf.argmax(self.source_pred_prob, 4) # predictions
        self.source_y_compact = tf.argmax(self.source_y, 4) # ground truth
        self.source_dice_eval_arr = self._eval_dice_during_train(self.source_y, self.source_pred_compact)

        # # segmentation for target
        self.target_pred_prob = pixel_wise_softmax_2(self.target_seg_logits)
        self.target_pred_compact = tf.argmax(self.target_pred_prob, 4) # predictions
        self.target_y_compact = tf.argmax(self.target_y, 4) # ground truth
        self.target_dice_eval_arr = self._eval_dice_during_train(self.target_y, self.target_pred_compact)


    def separate_network(self, input_data, is_training, keep_prob, feature_base=16, bn_scope=''):
        # input_data format as (batch, depth, height, width, channels)

        # down-sampling path
        # first level
        encoder1_1 = conv_bn_relu(inputs=input_data, output_channels=feature_base, kernel_size=3, stride=1, is_training=is_training, keep_prob=keep_prob, name='encoder1_1', bn_scope=bn_scope)
        encoder1_2 = conv_bn_relu(inputs=encoder1_1, output_channels=feature_base * 2, kernel_size=3, stride=1, is_training=is_training, keep_prob=keep_prob, name='encoder1_2', bn_scope=bn_scope)
        pool1 = tf.layers.max_pooling3d(inputs=encoder1_2, pool_size=2, strides=2, padding='same', name='pool1')
        # second level
        encoder2_1 = conv_bn_relu(inputs=pool1, output_channels=feature_base * 2, kernel_size=3, stride=1, is_training=is_training, keep_prob=keep_prob, name='encoder2_1', bn_scope=bn_scope)
        encoder2_2 = conv_bn_relu(inputs=encoder2_1, output_channels=feature_base * 4, kernel_size=3, stride=1, is_training=is_training, keep_prob=keep_prob, name='encoder2_2', bn_scope=bn_scope)
        pool2 = tf.layers.max_pooling3d(inputs=encoder2_2, pool_size=2, strides=2, padding='same', name='pool2')
        # third level
        encoder3_1 = conv_bn_relu(inputs=pool2, output_channels=feature_base * 4, kernel_size=[3,3,1], stride=1, is_training=is_training, keep_prob=keep_prob, name='encoder3_1', bn_scope=bn_scope)
        encoder3_2 = conv_bn_relu(inputs=encoder3_1, output_channels=feature_base * 8, kernel_size=[3,3,1], stride=1, is_training=is_training, keep_prob=keep_prob, name='encoder3_2', bn_scope=bn_scope)
        pool3 = tf.layers.max_pooling3d(inputs=encoder3_2, pool_size=2, strides=2, padding='same', name='pool3')
        # forth level
        encoder4_1 = conv_bn_relu(inputs=pool3, output_channels=feature_base * 8, kernel_size=[3,3,1], stride=1, is_training=is_training, keep_prob=keep_prob, name='encoder4_1', bn_scope=bn_scope)
        encoder4_2 = conv_bn_relu(inputs=encoder4_1, output_channels=feature_base * 16, kernel_size=[3,3,1], stride=1, is_training=is_training, keep_prob=keep_prob, name='encoder4_2', bn_scope=bn_scope)
        bottom = encoder4_2

        return [encoder4_2, encoder3_2, encoder2_2, encoder1_2]

    def joint_network(self, input_data, is_training, keep_prob, feature_base=16, bn_scope=''):
        # up-sampling path
        # third level
        bottom, encoder3_2, encoder2_2, encoder1_2 = input_data

        deconv3 = deconv_bn_relu(inputs=bottom, output_channels=feature_base * 8, is_training=is_training, kernel_size=[3,3,1], keep_prob=keep_prob, name='deconv3', bn_scope=bn_scope)
        concat_3 = tf.concat([deconv3, encoder3_2], axis=4, name='concat_3')
        decoder3_1 = conv_bn_relu(inputs=concat_3, output_channels=feature_base * 8, kernel_size=[3,3,1], stride=1, is_training=is_training, keep_prob=keep_prob, name='decoder3_1', bn_scope=bn_scope)
        decoder3_2 = conv_bn_relu(inputs=decoder3_1, output_channels=feature_base * 8, kernel_size=[3,3,1], stride=1, is_training=is_training, keep_prob=keep_prob, name='decoder3_2', bn_scope=bn_scope)
        # second level
        deconv2 = deconv_bn_relu(inputs=decoder3_2, output_channels=feature_base * 8, is_training=is_training, kernel_size=[3,3,1], keep_prob=keep_prob, name='deconv2', bn_scope=bn_scope)
        concat_2 = tf.concat([deconv2, encoder2_2], axis=4, name='concat_2')
        decoder2_1 = conv_bn_relu(inputs=concat_2, output_channels=feature_base * 4, kernel_size=3, stride=1, is_training=is_training, keep_prob=keep_prob, name='decoder2_1', bn_scope=bn_scope)
        decoder2_2 = conv_bn_relu(inputs=decoder2_1, output_channels=feature_base * 4, kernel_size=3, stride=1, is_training=is_training, keep_prob=keep_prob, name='decoder2_2', bn_scope=bn_scope)
        # first level
        deconv1 = deconv_bn_relu(inputs=decoder2_2, output_channels=feature_base * 4, is_training=is_training, kernel_size=[3,3,3] , keep_prob=keep_prob, name='deconv1', bn_scope=bn_scope)
        concat_1 = tf.concat([deconv1, encoder1_2], axis=4, name='concat_1')
        decoder1_1 = conv_bn_relu(inputs=concat_1, output_channels=feature_base * 2, kernel_size=3, stride=1,is_training=is_training, keep_prob=keep_prob, name='decoder1_1', bn_scope=bn_scope)
        decoder1_2 = conv_bn_relu(inputs=decoder1_1, output_channels=feature_base * 2, kernel_size=3, stride=1, is_training=is_training, keep_prob=keep_prob, name='decoder1_2', bn_scope=bn_scope)
        feature = decoder1_2
        # predicted probability
        logits = conv3d(inputs=feature, output_channels=self.num_class, kernel_size=1, stride=1, use_bias=True, name='seg_logits')

        return logits

    def _get_segmentation_cost(self, seg_logits, seg_gt, variables_list, cost_kwargs):
        """
        calculate the loss for segmentation prediction
        :param seg_logits: activations before the Softmax function
        :param seg_gt: ground truth segmentaiton mask
        :return: segmentation loss, according to the cost_kwargs setting, cross-entropy weighted loss and dice loss
        """
        softmaxpred = tf.nn.softmax(seg_logits)

        # calculate dice loss, - 2*interesction/union, with relaxed for gradients back-propagation
        dice = 0
        for i in xrange(self.num_class):
            inse = tf.reduce_sum(softmaxpred[:,:,:,:,i]*seg_gt[:,:,:,:,i])
            l = tf.reduce_sum(softmaxpred[:,:,:,:,i]*softmaxpred[:,:,:,:,i])
            r = tf.reduce_sum(seg_gt[:,:,:,:,i])
            dice += 2.0 * inse/(l+r+1e-7) # here 1e-7 is relaxation eps
        dice_loss = -1.0 * dice / self.num_class

        # calculate cross-entropy weighted loss
        ce_weighted = 0
        for i in xrange(self.num_class):
            gti = seg_gt[:,:,:,:,i]
            predi = softmaxpred[:,:,:,:,i]
            weighted = 1-(tf.reduce_sum(gti) / tf.reduce_sum(seg_gt))
            ce_weighted += -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))
        ce_weighted_loss = tf.reduce_mean(ce_weighted)

        L2_norm = sum([tf.nn.l2_loss(var) for var in variables_list])
        total_loss =  cost_kwargs["miu_seg_dice"] * dice_loss + cost_kwargs["miu_seg_ce"] * ce_weighted_loss + cost_kwargs["miu_seg_L2_norm"] * L2_norm

        return dice_loss, ce_weighted_loss, total_loss

    def _get_kd_cost(self, source_logits, source_gt, target_logits, target_gt):

        kd_loss = 0.0

        self.source_logits = source_logits
        self.target_logits = target_logits

        self.source_prob = []
        self.target_prob = []

        temperature = 2.0

        for i in xrange(self.num_class):
            eps = 1e-6

            self.s_mask = tf.tile(tf.expand_dims(source_gt[:,:,:,:,i], -1), [1,1,1,1,self.num_class])
            self.s_logits_mask_out = self.source_logits * self.s_mask
            self.s_logits_avg = tf.reduce_sum(self.s_logits_mask_out, [0,1,2,3]) / (tf.reduce_sum(source_gt[:,:,:,:,i]) + eps)
            self.s_soft_prob = tf.nn.softmax(self.s_logits_avg/temperature)
            self.source_prob.append(self.s_soft_prob)

            self.t_mask = tf.tile(tf.expand_dims(target_gt[:,:,:,:,i], -1), [1,1,1,1,self.num_class])
            self.t_logits_mask_out = self.target_logits * self.t_mask
            self.t_logits_avg = tf.reduce_sum(self.t_logits_mask_out, [0,1,2,3]) / (tf.reduce_sum(target_gt[:,:,:,:,i]) + eps)
            self.t_soft_prob = tf.nn.softmax(self.t_logits_avg/temperature)

            self.target_prob.append(self.t_soft_prob)

            ## KL divergence loss
            loss = (tf.reduce_sum(self.s_soft_prob * tf.log(self.s_soft_prob/self.t_soft_prob)) + tf.reduce_sum(self.t_soft_prob * tf.log(self.t_soft_prob/self.s_soft_prob))) / 2.0

            ## Original knowledge distilling
            ## here use the function softmax_cross_entropy_with_logits_v2, both logits and labels back-propagate gradients
            ## logits MUST input logits which have not been softmax, as this function self-include softmax)
            # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.s_logits_avg, labels=self.t_soft_prob)

            ## L2 Norm
            # loss = tf.nn.l2_loss(self.s_soft_prob - self.t_soft_prob) / self.n_class

            kd_loss += loss

        self.kd_loss = kd_loss / self.num_class

        return self.kd_loss

    def _get_L2_Norm(self, variables_list):

        L2_norm = tf.reduce_sum([tf.nn.l2_loss(var) for var in variables_list])
        return L2_norm


    def _get_recon_cost_L1(self, recon, image):
        #return tf.reduce_mean(tf.abs(recon-image))
        return tf.reduce_sum(tf.nn.l2_loss(recon - image))

    def _eval_dice_during_train(self, labels, compact_pred):
        """
        calculate standard dice for evaluation, here uses the class prediction, not the probability
        """
        dice_arr = []
        eps = 1e-7
        pred = tf.one_hot(compact_pred, depth = self.num_class, axis = -1)
        for i in xrange(self.num_class):
            inse = tf.reduce_sum(pred[:,:,:,:,i] * labels[:,:,:,:,i])
            union = tf.reduce_sum(pred[:,:,:,:,i]) + tf.reduce_sum(labels[:,:,:,:,i])
            dice_arr.append(2.0 * inse / (union + eps))

        return dice_arr