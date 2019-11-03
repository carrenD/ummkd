import tensorflow as tf
# from tensorflow.python import debug as tf_debug
import numpy as np
import os
import glob
import logging
import datetime
from lib.util import _read_lists

from network import Build_network as build_network
from train_combine import Trainer

logging.basicConfig(filename="general_log", level=logging.DEBUG)

currtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

## train segmentation CNN for MRI & CT
## One modality is treated as source, the other modality is treated as target. Order doesn't matter

source_train_fid = "/data/projects/multimodal_learning/data/tftrain/mr_train_list"
source_val_fid = "/data/projects/multimodal_learning/data/tftrain/mr_val_list"

target_train_fid = "/data/projects/multimodal_learning/data/tftrain/ct_train_list"
target_val_fid = "/data/projects/multimodal_learning/data/tftrain/ct_val_list"

cost_kwargs = {
    "miu_seg_dice": 1.0,
    "miu_seg_ce": 1.0,
    "miu_seg_L2_norm": 1e-4,
    "miu_kd": 0.5,
}

network_config = { # set whether generator/discriminator trainable
    "source_kernel_update": True, # flag in tf.Variable(w, trainable= ) initialization for convolution kernels, whether collect this in trainable variable
    "target_kernel_update": True,
    "joint_kernel_update" : True,
    "discriminator_kernel_update": True,
    "source_bn_update": True, # flag in tf.contrib.layers.batch_norm(input_feature, trainable= ) initialization for bn, whether collect this in trainable variable
    "target_bn_update": True,
    "joint_bn_update": True,
    "discriminator_bn_update": True,
}

opt_kwargs = {
    "learning_rate": 1e-3,
    "prefix": "multimodal-bn-kd-KL",
    "port": "6008",
}


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

base_path = "/data/projects/multimodal_learning/exp_results/"
output_path = base_path + opt_kwargs["prefix"]


def main(restored_model=None, test_model=None, phase='training'):

    source_train_list = _read_lists(source_train_fid)
    source_val_list = _read_lists(source_val_fid)
    target_train_list = _read_lists(target_train_fid)
    target_val_list = _read_lists(target_val_fid)

    network = build_network(channels=3, n_class=5, batch_size=8, cost_kwargs=cost_kwargs, network_config=network_config, opt_kwargs=opt_kwargs)

    logging.info("Network built")

    trainer = Trainer(net = network,\
                      source_train_list = source_train_list,\
                      source_val_list = source_val_list,\
                      target_train_list = target_train_list,\
                      target_val_list = target_val_list, \
                      output_path = output_path, \
                      opt_kwargs = opt_kwargs,\
                      num_epochs = 750,\
                      checkpoint_space = 300)

    # start tensorboard before getting started
    command2 = "fuser " + opt_kwargs["port"] + "/tcp -k"
    os.system(command2)
    command1 = "tensorboard --logdir=" + output_path + " --port=" + opt_kwargs["port"] + " &"
    os.system(command1)

    if phase == 'training':
        trainer.train_segmenter(restored_model = restored_model)

    if phase == 'testing' 
        # here are for the testing phase
        test_list_fid = "/data/projects/multimodal_learning/data/npz_mr_test_5cls"
        test_nii_list_fid = "/data/projects/multimodal_learning/data/test_mr_nii_list"
        part = "source"
        logging.info('performance on source ...')
        source_dice = trainer.test(test_model = test_model, part = part, test_list_fid = test_list_fid, test_nii_list_fid = test_nii_list_fid)


        test_list_fid = "/data/projects/multimodal_learning/data/npz_ct_test_5cls"
        test_nii_list_fid = "/data/projects/multimodal_learning/data/test_ct_nii_list"
        part = "target"
        logging.info('performance on target ...')
        target_dice = trainer.test(test_model = test_model, part = part, test_list_fid = test_list_fid, test_nii_list_fid = test_nii_list_fid)

        return source_dice, target_dice

if __name__ == "__main__":

    # for training, can specify a restore checkpoint
    restored_model = None
    main(restored_model = restored_model, phase='training')

    # # for testing, need to specify a model to be tested
    # test_model = '/path/to/test_model.cpkt'
    # source_dice, target_dice = main(test_model=test_model, phase='testing')