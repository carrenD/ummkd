import os
import time
import numpy as np
import __future__
import logging
import tensorflow as tf
# import pymedimage.visualize as viz
# import pymedimage.niftiio as nio
from lib.util import _label_decomp, _eval_dice, _read_lists, _save_nii


raw_size = [256, 256, 3] # original raw input size
volume_size = [256, 256, 3] # volume size after processing, for the tfrecord file
label_size = [256, 256, 1] # size of label

decomp_feature = { # configuration for decoding tf_record file
            'dsize_dim0': tf.FixedLenFeature([], tf.int64),
            'dsize_dim1': tf.FixedLenFeature([], tf.int64),
            'dsize_dim2': tf.FixedLenFeature([], tf.int64),
            'lsize_dim0': tf.FixedLenFeature([], tf.int64),
            'lsize_dim1': tf.FixedLenFeature([], tf.int64),
            'lsize_dim2': tf.FixedLenFeature([], tf.int64),
            'data_vol': tf.FixedLenFeature([], tf.string),
            'label_vol': tf.FixedLenFeature([], tf.string)}

class_map = {  # a map used for mapping label value to its name, used for output
    "0": "bg",
    "1": "lv_myo",
    "2": "la_blood",
    "3": "lv_blood",
    "4": "aa"
}


class Trainer(object):

    def __init__(self, net, source_train_list, source_val_list, target_train_list, target_val_list, output_path, \
                 opt_kwargs=None, num_epochs = 1000, training_iters = 200, checkpoint_space = 200, lr_update_flag = False):

        self.net = net
        self.checkpoint_space = checkpoint_space # intervals between saving a checkpoint and decaying learning rate
        self.opt_kwargs = opt_kwargs

        self.num_epochs = num_epochs
        self.training_iters = training_iters

        self.source_train_list = source_train_list
        self.source_val_list = source_val_list
        self.target_train_list = target_train_list
        self.target_val_list = target_val_list

        self.source_train_queue = tf.train.string_input_producer(self.source_train_list, num_epochs = None, shuffle = True) # tensorflow input queue for CT supervision (disabled), CT and MRI
        self.source_val_queue = tf.train.string_input_producer(self.source_val_list, num_epochs = None, shuffle = True)
        self.target_train_queue = tf.train.string_input_producer(self.target_train_list, num_epochs = None, shuffle = True)
        self.target_val_queue = tf.train.string_input_producer(self.target_val_list, num_epochs = None, shuffle = True)

        self.lr_update_flag = lr_update_flag # if true, manually update learning rate before running

        self.output_path = output_path
        if not os.path.exists(self.output_path):
            logging.info("Allocating '{:}'".format(self.output_path))
            os.makedirs(self.output_path)


    def _get_optimizers(self):

        self.global_step = tf.Variable(0, name = "global_step")

        self.learning_rate = self.opt_kwargs["learning_rate"]
        self.learning_rate_node = tf.Variable(self.learning_rate, name = "learning_rate")

        # optimizer for source segmentation CNN
        optimizer_overall = tf.train.AdamOptimizer(learning_rate = self.learning_rate_node).minimize(
                                                                       loss = self.net.overall_loss, \
                                                                       var_list = self.net.var_list, \
                                                                       global_step = self.global_step) # here var_list include all kernel, bn beta, bn gamma
        return optimizer_overall


    def restore_model(self, sess, restored_model):

        if restored_model is not None:

            print 'restoring model ....'

            saver = tf.train.Saver()
            saver.restore(sess, restored_model)
            logging.info("Fine tune the segmenter, model restored from %s" % restored_model)
        else:
            logging.info("Training the segmenter model from scratch")


    def train_segmenter(self, restored_model, display_step=1):

        print "Start training the segmenter ..."

        self.optimizer_overall = self._get_optimizers()
        self._init_tfboard()

        init_glb = tf.global_variables_initializer()
        init_loc = tf.variables_initializer(tf.local_variables())
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        with open(os.path.join(self.output_path, 'eva.txt'), 'w') as f:
            f.write("Record the test performance on the fly as training ...\n")

        with tf.Session(config=config) as sess:
            sess.run([init_glb, init_loc])
            coord = tf.train.Coordinator()

            train_summary_writer = tf.summary.FileWriter(self.output_path + "/train_log_"  + self.opt_kwargs["prefix"], graph=sess.graph)
            val_summary_writer = tf.summary.FileWriter(self.output_path + "/val_log_" + self.opt_kwargs["prefix"], graph=sess.graph)

            self.restore_model(sess, restored_model)

            source_train_feed, source_train_feed_fid = self.next_batch(self.source_train_queue)
            source_val_feed, source_val_feed_fid = self.next_batch(self.source_val_queue)
            target_train_feed, target_train_feed_fid = self.next_batch(self.target_train_queue)
            target_val_feed, target_val_feed_fid = self.next_batch(self.target_val_queue)
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)

            best_avg = 0
            best_model_performance = []
            best_model_save_path = None

            for epoch in xrange(self.num_epochs):
                for step in xrange((epoch*self.training_iters), ((epoch+1)*self.training_iters)):
                    logging.info("Running step %s epoch %s ..."%(str(step), str(epoch)))
                    start = time.time()

                    source_train_batch, source_train_fid = sess.run([source_train_feed, source_train_feed_fid])
                    source_train_batch_x = source_train_batch[:, :, :, 0:3]
                    source_train_batch_y = _label_decomp(source_train_batch[:, :, :, 3], self.net.n_class)

                    target_train_batch, target_train_fid = sess.run([target_train_feed, target_train_feed_fid])
                    target_train_batch_x = target_train_batch[:, :, :, 0:3]
                    target_train_batch_y = _label_decomp(target_train_batch[:, :, :, 3], self.net.n_class)


                    _, source_loss, target_loss, kd_loss, source_prob, target_prob, lr = sess.run(\
                              (self.optimizer_overall, self.net.source_seg_dice_loss, self.net.target_seg_dice_loss,\
                               self.net.kd_loss, self.net.source_prob, self.net.target_prob, self.learning_rate_node),\
                                  feed_dict={self.net.source: source_train_batch_x,
                                             self.net.source_y: source_train_batch_y,
                                             self.net.target: target_train_batch_x,
                                             self.net.target_y: target_train_batch_y,
                                             self.net.keep_prob: 0.75,})

                    logging.info("Training at global step %s epoch %s, source loss is %0.4f, target loss is %0.4f"%(str(self.global_step.eval()), str(epoch), source_loss, target_loss))
                    logging.info("Knowledge Distilling loss: %0.4f" % kd_loss)
                    print "source prob:", source_prob
                    print "target prob:", target_prob
                    logging.info("Current learning rate %0.8f" % lr)
                    logging.info("Time elapsed %s seconds"%(str(time.time() - start)))

                    if step % (display_step * 20) == 0:
                        print 'update the tensorboard for training ...'
                        self.minibatch_stats_segmenter(sess, train_summary_writer, step, source_train_batch_x, source_train_batch_y, target_train_batch_x, target_train_batch_y, section = "train")

                    if step % (display_step * 20) == 0:
                        print 'update the tensorboard for validation ...'
                        source_val_batch = source_val_feed.eval()
                        source_val_batch_x = source_val_batch[:, :, :, 0:3]
                        source_val_batch_y = _label_decomp(source_val_batch[:, :, :, 3], self.net.n_class)

                        target_val_batch = target_val_feed.eval()
                        target_val_batch_x = target_val_batch[:, :, :, 0:3]
                        target_val_batch_y = _label_decomp(target_val_batch[:, :, :, 3], self.net.n_class)

                        self.minibatch_stats_segmenter(sess, val_summary_writer, step, source_val_batch_x, source_val_batch_y, target_val_batch_x, target_val_batch_y, section="val")


                    ## The followings are learning rate decay
                    if self.global_step.eval() % 1000 == 0:
                        _pre_lr = sess.run(self.learning_rate_node)
                        sess.run(tf.assign(self.learning_rate_node, _pre_lr * 0.95))


                    # save the model periodically
                    if self.global_step.eval() % (self.checkpoint_space) == 0:
                        saver = tf.train.Saver()
                        saved_model_name = self.opt_kwargs["prefix"] + "_itr%d_model.cpkt" % self.global_step.eval()
                        save_path = saver.save(sess, os.path.join(self.output_path, saved_model_name), global_step = self.global_step.eval())
                        logging.info("Model saved as step %d, save path is %s" % (self.global_step.eval(), save_path))


            logging.info("Modeling training Finished!")
            coord.request_stop()
            coord.join(threads)
            return 0


    def test(self, test_model, part, test_list_fid, test_nii_list_fid):

        test_list = _read_lists(test_list_fid)
        test_nii_list = _read_lists(test_nii_list_fid)
        test_pair_list = zip(test_list, test_nii_list)

        init_glb = tf.global_variables_initializer()
        init_loc = tf.variables_initializer(tf.local_variables())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run([init_glb, init_loc])
            coord = tf.train.Coordinator()

            saver = tf.train.Saver()
            saver.restore(sess, test_model)
            logging.info("test segmenter, model is %s" % test_model)

            dice = []
            for idx_file, pair in enumerate(test_pair_list):
                fid = pair[0]  # this is npz data
                _npz_dict = np.load(fid)
                raw = np.flip(np.flip(_npz_dict['arr_0'], axis=0), axis=1)
                gt_y = np.flip(np.flip(_npz_dict['arr_1'], axis=0), axis=1)
                pred_y = np.zeros(gt_y.shape)

                frame_list = [kk for kk in range(1, raw.shape[2] - 1)]
                np.random.shuffle(frame_list)
                for ii in xrange(int(np.floor(raw.shape[2] // self.net.batch_size))):
                    vol = np.zeros([self.net.batch_size, raw_size[0], raw_size[1], raw_size[2]])
                    for idx, jj in enumerate(frame_list[ii * self.net.batch_size: (ii + 1) * self.net.batch_size]):
                        vol[idx, ...] = raw[..., jj - 1: jj + 2].copy()

                    if part == "source":
                        pred = sess.run(self.net.source_pred_compact, feed_dict={self.net.source: vol,
                                                                                 self.net.keep_prob: 1.0,
                                                                                 self.net.training_mode_source: False,
                                                                                 self.net.training_mode_target: False,})
                    elif part == "target":
                        pred = sess.run(self.net.target_pred_compact, feed_dict={self.net.target: vol,
                                                                                 self.net.keep_prob: 1.0,
                                                                                 self.net.training_mode_source: False,
                                                                                 self.net.training_mode_target: False,})

                    for idx, jj in enumerate(frame_list[ii * self.net.batch_size: (ii + 1) * self.net.batch_size]):
                        pred_y[..., jj] = pred[idx, ...].copy()

                dice_subject = _eval_dice(gt_y, pred_y)
                dice.append(dice_subject)
                _save_nii(pred_y, gt_y, pair[1], self.output_path)

            print dice

            dice_avg = np.mean(dice, axis=0).tolist()
            dice_std = np.std(dice, axis=0).tolist()

            for cls in xrange(1, self.net.n_class):
                logging.info("%s avg dice is %.4f, std is %.4f" % (class_map[str(cls)], dice_avg[cls-1], dice_std[cls-1]))
            logging.info("average dice is: %f" % np.mean(dice_avg))
            

        return dice_avg


    def _init_tfboard(self):
        """
        initialization and tensorboard summary
        """

        scalar_summaries = []
        train_images = []
        val_images = []

        scalar_summaries.append(tf.summary.scalar('learning_rate', self.learning_rate_node))
        scalar_summaries.append(tf.summary.scalar("source_segmentation_dice", self.net.source_seg_dice_loss))
        scalar_summaries.append(tf.summary.scalar("source_segmentation_ce", self.net.source_seg_ce_loss))

        scalar_summaries.append(tf.summary.scalar('source_segmentation_dice_' + class_map["1"], self.net.source_dice_eval_arr[1]))
        scalar_summaries.append(tf.summary.scalar('source_segmentation_dice_' + class_map["2"], self.net.source_dice_eval_arr[2]))
        scalar_summaries.append(tf.summary.scalar('source_segmentation_dice_' + class_map["3"], self.net.source_dice_eval_arr[3]))
        scalar_summaries.append(tf.summary.scalar('source_segmentation_dice_' + class_map["4"], self.net.source_dice_eval_arr[4]))

        train_images.append(tf.summary.image("source_pred_train", tf.expand_dims(tf.cast(self.net.source_pred_compact, tf.float32), 3)))
        train_images.append(tf.summary.image('source_image_train', tf.expand_dims(tf.cast(self.net.source[:,:,:,1], tf.float32), 3)))
        train_images.append(tf.summary.image('source_gt_train', tf.expand_dims(tf.cast(self.net.source_y_compact, tf.float32), 3)))

        val_images.append(tf.summary.image("source_pred_val", tf.expand_dims(tf.cast(self.net.source_pred_compact, tf.float32), 3)))
        val_images.append(tf.summary.image('source_image_val', tf.expand_dims(tf.cast(self.net.source[:,:,:,1], tf.float32), 3)))
        val_images.append(tf.summary.image('source_gt_val', tf.expand_dims(tf.cast(self.net.source_y_compact, tf.float32), 3)))

        scalar_summaries.append(tf.summary.scalar("target_segmentation_dice", self.net.target_seg_dice_loss))
        scalar_summaries.append(tf.summary.scalar("target_segmentation_ce", self.net.target_seg_ce_loss))

        scalar_summaries.append(tf.summary.scalar('target_segmentation_dice_c1_lv_myo', self.net.target_dice_eval_arr[1]))
        scalar_summaries.append(tf.summary.scalar('target_segmentation_dice_c2_la_blood', self.net.target_dice_eval_arr[2]))
        scalar_summaries.append(tf.summary.scalar('target_segmentation_dice_c3_lv_blood', self.net.target_dice_eval_arr[3]))
        scalar_summaries.append(tf.summary.scalar('target_segmentation_dice_c4_aa', self.net.target_dice_eval_arr[4]))

        scalar_summaries.append(tf.summary.scalar('kd_loss', self.net.kd_loss))

        self.scalar_summary_op = tf.summary.merge(scalar_summaries)
        self.train_image_summary_op = tf.summary.merge(train_images)
        self.val_image_summary_op = tf.summary.merge(val_images)


    def minibatch_stats_segmenter(self, sess, summary_writer, step, source_batch_x, source_batch_y, target_batch_x, target_batch_y, section):

        if section == 'train':
            summary_str, summary_img = sess.run([self.scalar_summary_op, self.train_image_summary_op],
                                                feed_dict={self.net.source: source_batch_x,
                                                           self.net.source_y: source_batch_y,
                                                           self.net.target: target_batch_x,
                                                           self.net.target_y: target_batch_y,
                                                           self.net.training_mode_source: False,
                                                           self.net.training_mode_target: False,
                                                           self.net.keep_prob: 1.})
            summary_writer.add_summary(summary_str, step)
            summary_writer.add_summary(summary_img, step)
            summary_writer.flush()

        elif section == 'val':
            summary_str, summary_img = sess.run([self.scalar_summary_op, self.val_image_summary_op],
                                                feed_dict={self.net.source: source_batch_x,
                                                           self.net.source_y: source_batch_y,
                                                           self.net.target: target_batch_x,
                                                           self.net.target_y: target_batch_y,
                                                           self.net.training_mode_source: False,
                                                           self.net.training_mode_target: False,
                                                           self.net.keep_prob: 1.})
            summary_writer.add_summary(summary_str, step)
            summary_writer.add_summary(summary_img, step)
            summary_writer.flush()


    def next_batch(self, input_queue, capacity = 120, num_threads = 4, min_after_dequeue = 30, label_type = 'float'):
        """ move original input pipeline here"""
        reader = tf.TFRecordReader()
        fid, serialized_example = reader.read(input_queue)
        parser = tf.parse_single_example(serialized_example, features = decomp_feature)
        dsize_dim0 = tf.cast(parser['dsize_dim0'], tf.int32)
        dsize_dim1 = tf.cast(parser['dsize_dim1'], tf.int32)
        dsize_dim2 = tf.cast(parser['dsize_dim2'], tf.int32)
        lsize_dim0 = tf.cast(parser['lsize_dim0'], tf.int32)
        lsize_dim1 = tf.cast(parser['lsize_dim1'], tf.int32)
        lsize_dim2 = tf.cast(parser['lsize_dim2'], tf.int32)
        data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
        label_vol = tf.decode_raw(parser['label_vol'], tf.float32)

        data_vol = tf.reshape(data_vol, raw_size)
        label_vol = tf.reshape(label_vol, raw_size)
        data_vol = tf.slice(data_vol, [0,0,0], volume_size)
        label_vol = tf.slice(label_vol, [0,0,1], label_size)

        data_feed, label_feed, fid_feed = tf.train.shuffle_batch([data_vol, label_vol, fid], batch_size =self.net.batch_size , capacity = capacity, \
                                                            num_threads = num_threads, min_after_dequeue = min_after_dequeue)

        pair_feed = tf.concat([data_feed, label_feed], axis = 3)

        return pair_feed, fid_feed