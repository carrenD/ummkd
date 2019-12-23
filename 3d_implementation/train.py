import os
import time
import numpy as np
import logging
import tensorflow as tf
import data_loader


class_map = {  # a map used for mapping label value to its name, used for output
    "0": "bg",
    "1": "liver",
    "2": "right kidney",
    "3": "left kidney",
    "4": "spleen" }

class Trainer(object):

    def __init__(self, net, source_train_list, target_train_list, source_val_list, target_val_list, output_path, train_config={}, version_config={}):

        self.net = net

        self.source_train_list = source_train_list
        self.target_train_list = target_train_list
        self.source_val_list = source_val_list
        self.target_val_list = target_val_list

        self.output_path = output_path

        self.train_config = train_config
        self.batch_size = self.train_config["batch_size"]
        self.num_epochs = self.train_config["num_epochs"]
        self.training_iters = self.train_config["training_iters"]
        self.checkpoint_space = self.train_config["checkpoint_space"]  # intervals between saving a checkpoint and decaying learning rate

        self.version_config = version_config

    def _get_optimizers(self):

        self.global_step = tf.Variable(0, name = "global_step")

        self.learning_rate = self.train_config["learning_rate"]
        self.learning_rate_node = tf.Variable(self.learning_rate, name = "learning_rate")

        # Here is for batch normalization, moving mean and moving variance work properly
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer_overall_segmentation = tf.train.AdamOptimizer(learning_rate = self.learning_rate_node).minimize(
                                                                           loss = self.net.overall_loss, \
                                                                           var_list = self.net.joint_variables, \
                                                                           global_step = self.global_step)

        return optimizer_overall_segmentation


    def restore_model(self, sess, restored_model):

        if restored_model is not None:
            saver = tf.train.Saver()
            saver.restore(sess, restored_model)
            logging.info("Fine tune the segmenter, model restored from %s" % restored_model)
        else:
            logging.info("Training the segmenter model from scratch")


    def train_segmenter(self, restored_model=None):

        print "Start training the segmenter ..."

        # with open(os.path.join(self.output_path, 'eva.txt'), 'w') as f:
        #     f.write("Record the test performance on the fly as training ...\n")

        self.source_inputs_train = data_loader._load_data(datalist=self.source_train_list, patch_size=self.net.patch_size, batch_size=self.batch_size, num_class=self.net.num_class)
        self.target_inputs_train = data_loader._load_data(datalist=self.target_train_list, patch_size=self.net.patch_size, batch_size=self.batch_size, num_class=self.net.num_class)
        self.source_inputs_val = data_loader._load_data(datalist=self.source_val_list, patch_size=self.net.patch_size, batch_size=self.batch_size, num_class=self.net.num_class)
        self.target_inputs_val = data_loader._load_data(datalist=self.target_val_list, patch_size=self.net.patch_size, batch_size=self.batch_size, num_class=self.net.num_class)

        self.optimizer_overall= self._get_optimizers()
        self._init_tfboard()

        # config = tf.ConfigProto(log_device_placement=False)
        # config.gpu_options.allow_growth = False
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            if restored_model is not None:
                self.restore_model(sess, restored_model)
            else:
        
                init_glb = tf.global_variables_initializer()
                init_loc = tf.variables_initializer(tf.local_variables())
                sess.run([init_glb, init_loc])

    
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    
            train_summary_writer = tf.summary.FileWriter(self.output_path + "/train_log_"  + self.version_config["prefix"], graph=sess.graph)
            val_summary_writer = tf.summary.FileWriter(self.output_path + "/val_log_" + self.version_config["prefix"], graph=sess.graph)
    
            for epoch in xrange(self.num_epochs):
                for step in xrange((epoch*self.training_iters), ((epoch+1)*self.training_iters)):
            
                    logging.info("Running step %s epoch %s ..."%(str(step), str(epoch)))
            
                    start = time.time()

                    source_train_x, source_train_y = sess.run(self.source_inputs_train)
                    target_train_x, target_train_y = sess.run(self.target_inputs_train)

                    _, source_loss, target_loss, kd_loss, lr= sess.run((self.optimizer_overall, self.net.source_seg_dice_loss, self.net.target_seg_dice_loss, self.net.weighted_kd_loss, self.learning_rate_node),\
                                                                    feed_dict={self.net.source: source_train_x,
                                                                               self.net.source_y: source_train_y,
                                                                               self.net.target: target_train_x,
                                                                               self.net.target_y: target_train_y,
                                                                               self.net.training_mode: True,
                                                                               self.net.keep_prob: 0.75,
                                                                               })


                    logging.info("Global step %s epoch %s, source loss is %0.4f, targtet loss is %0.4f, kd_loss is %0.4f, lr is %.8f"%(str(self.global_step.eval()), str(epoch), source_loss, target_loss, kd_loss, lr))
                    logging.info("Time elapsed %s seconds"%(str(time.time() - start)))
                    # for i in range(5):
                    #     print class_map[str(i)], dice_array[i]
                    if self.global_step.eval() % (self.train_config["display_step"]) == 0:
                        #logging.info("update the tensorboard for training ...")
                        self.minibatch_stats_segmenter_source(sess, train_summary_writer, step, source_train_x, source_train_y, target_train_x, target_train_y)
                        self.minibatch_stats_segmenter_target(sess, train_summary_writer, step, target_train_x, target_train_y)

                    if self.global_step.eval() % (100) == 0:
                        #logging.info("update the tensorboard for validation ...")
                        source_val_x, source_val_y = sess.run(self.source_inputs_val)
                        target_val_x, target_val_y = sess.run(self.target_inputs_val)
                        #print 'size val_x, size val_y: ', type(val_x), val_x.shape, val_y.shape
                        self.minibatch_stats_segmenter_source(sess, val_summary_writer, step, source_val_x, source_val_y, target_val_x, target_val_y, mode='val')
                        self.minibatch_stats_segmenter_target(sess, val_summary_writer, step, target_val_x, target_val_y, mode='val')

                    ## The followings are learning rate decay
                    if self.global_step.eval() % self.train_config["training_iters"] == 0:
                        _pre_lr = sess.run(self.learning_rate_node)
                        sess.run(tf.assign(self.learning_rate_node, _pre_lr * 0.95))
                        #logging.info("Learning rate decayed from %0.8f to %0.8f"%(_pre_lr, sess.run(self.learning_rate_node)))

                    # The followings are model saving for debug
                    if self.global_step.eval() % self.train_config["checkpoint_space"] == 0:
                        saver = tf.train.Saver()
                        saved_model_name = self.version_config["prefix"] + "_model.cpkt"
                        save_path = saver.save(sess, os.path.join(self.output_path + '/checkpoint', saved_model_name), global_step = self.global_step.eval())
                        logging.info("Model saved as step %d, save path is %s" % (self.global_step.eval(), save_path))


            logging.info("Modeling training Finished!")
            coord.request_stop()
            coord.join(threads)
            return 0

    def _init_tfboard(self):
        """
        initialization and tensorboard summary
        """
        source_scalar_summaries = []
        source_images = []

        target_scalar_summaries = []
        target_images = []

        source_scalar_summaries.append(tf.summary.scalar('learning_rate', self.learning_rate_node))
        source_scalar_summaries.append(tf.summary.scalar("source_dice", self.net.source_seg_dice_loss))
        source_scalar_summaries.append(tf.summary.scalar("kd_loss", self.net.kd_loss))

        source_images.append(tf.summary.image("source_image", tf.cast(tf.expand_dims(self.net.source[:,:,:,self.net.patch_size[2]//2,0], axis=3), tf.float32)))
        source_images.append(tf.summary.image('source_seg_gt', tf.cast(tf.expand_dims(self.net.source_y_compact[:,:,:,self.net.patch_size[2]//2], axis=3), tf.float32)))
        source_images.append(tf.summary.image('source_seg_prob', tf.cast(tf.expand_dims(self.net.source_pred_prob[:,:,:,self.net.patch_size[2]//2,3], axis=3), tf.float32)))
        source_images.append(tf.summary.image('source_seg_pred', tf.cast(tf.expand_dims(self.net.source_pred_compact[:,:,:,self.net.patch_size[2]//2], axis=3), tf.float32)))
        source_scalar_summaries.append(tf.summary.scalar('source_dice_' + class_map["0"], self.net.source_dice_eval_arr[0]))
        source_scalar_summaries.append(tf.summary.scalar('source_dice_' + class_map["1"], self.net.source_dice_eval_arr[1]))
        source_scalar_summaries.append(tf.summary.scalar('source_dice_' + class_map["2"], self.net.source_dice_eval_arr[2]))
        source_scalar_summaries.append(tf.summary.scalar('source_dice_' + class_map["3"], self.net.source_dice_eval_arr[3]))
        source_scalar_summaries.append(tf.summary.scalar('source_dice_' + class_map["4"], self.net.source_dice_eval_arr[4]))

        target_scalar_summaries.append(tf.summary.scalar("target_dice", self.net.target_seg_dice_loss))

        target_images.append(tf.summary.image("target_image", tf.cast(tf.expand_dims(self.net.target[:,:,:,self.net.patch_size[2]//2,0], axis=3), tf.float32)))
        target_images.append(tf.summary.image('target_seg_gt', tf.cast(tf.expand_dims(self.net.target_y_compact[:,:,:,self.net.patch_size[2]//2], axis=3), tf.float32)))
        target_images.append(tf.summary.image('target_seg_prob', tf.cast(tf.expand_dims(self.net.target_pred_prob[:,:,:,self.net.patch_size[2]//2,3], axis=3), tf.float32)))
        target_images.append(tf.summary.image('target_seg_pred', tf.cast(tf.expand_dims(self.net.target_pred_compact[:,:,:,self.net.patch_size[2]//2], axis=3), tf.float32)))
        target_scalar_summaries.append(tf.summary.scalar('target_dice_' + class_map["0"], self.net.target_dice_eval_arr[0]))
        target_scalar_summaries.append(tf.summary.scalar('target_dice_' + class_map["1"], self.net.target_dice_eval_arr[1]))
        target_scalar_summaries.append(tf.summary.scalar('target_dice_' + class_map["2"], self.net.target_dice_eval_arr[2]))
        target_scalar_summaries.append(tf.summary.scalar('target_dice_' + class_map["3"], self.net.target_dice_eval_arr[3]))
        target_scalar_summaries.append(tf.summary.scalar('target_dice_' + class_map["4"], self.net.target_dice_eval_arr[4]))

        self.source_scalar_summary_op = tf.summary.merge(source_scalar_summaries)
        self.source_image_summary_op = tf.summary.merge(source_images)
        self.target_scalar_summary_op = tf.summary.merge(target_scalar_summaries)
        self.target_image_summary_op = tf.summary.merge(target_images)


    def minibatch_stats_segmenter_source(self, sess, summary_writer, step, data_x, data_y, target_train_x, target_train_y, mode='train'):

        summary_str, summary_img = sess.run([self.source_scalar_summary_op, self.source_image_summary_op],
                                            feed_dict={self.net.source: data_x,
                                                       self.net.source_y: data_y,
                                                       self.net.target: target_train_x,
                                                       self.net.target_y: target_train_y,
                                                       self.net.training_mode: False,
                                                       self.net.keep_prob: 0.75
                                                       })
        summary_writer.add_summary(summary_str, step)
        summary_writer.add_summary(summary_img, step)
        summary_writer.flush()

    def minibatch_stats_segmenter_target(self, sess, summary_writer, step, data_x, data_y, mode='train'):

        summary_str, summary_img = sess.run([self.target_scalar_summary_op, self.target_image_summary_op],
                                            feed_dict={self.net.target: data_x,
                                                       self.net.target_y: data_y,
                                                       self.net.training_mode: False,
                                                       self.net.keep_prob: 0.75
                                                       })
        summary_writer.add_summary(summary_str, step)
        summary_writer.add_summary(summary_img, step)
        summary_writer.flush()
