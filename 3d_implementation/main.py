import tensorflow as tf
import numpy as np
import os
import sys
import logging
import datetime

from network import Build_network
from train import Trainer


if not str(sys.argv[1]):
    print('No GPU given... setting to 0')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1])


hyperparam_config = {
    "patch_size": [256,256,8],
    "input_channels": 1,
    "num_class": 5,
    "feature_base": 16,
}

train_config = {
    "batch_size": 2,
    "learning_rate": 1e-4,
    "num_epochs": 100,
    "training_iters": 500,
    "checkpoint_space": 200,
    "display_step":20,
}

version_config = {
    "prefix": 'sifn',
    "date": '1102'
}

cost_kwargs = {
    "miu_seg_dice": 1.0,
    "miu_seg_ce": 1.0,
    "miu_seg_L2_norm": 1e-4,
    "miu_kd": 0.0,
}

logging.basicConfig(filename='./output/log/'+version_config["prefix"] + "_" + version_config["date"]+'_log', level=logging.DEBUG, format='%(asctime)s %(message)s')
currtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
logging.getLogger().addHandler(logging.StreamHandler())
def main(restored_model=None):

    # base_path = "/data/projects/test_specific"
    base_path = './'

    source_training_list = base_path + "/data/mr_train_list"
    target_training_list = base_path + "/data/ben_ct_train_list"
    source_validation_list = base_path + "/data/mr_test_list"
    target_validation_list = base_path + "/data/ben_ct_test_list"
    # test_list = base_path + "/data/test_list.txt"

    output_path = base_path + "/output/" + version_config["prefix"] + "_" + version_config["date"]
    if not os.path.exists(output_path+'/checkpoint'):
        logging.info("Allocating '{:}'".format(output_path))
        os.makedirs(output_path+'/checkpoint')

    os.system('cp main_joint_bn_kd.py %s' % (output_path)) # bkp of train procedure
    os.system('cp train_joint_bn_kd_one.py %s' % (output_path)) # bkp of train procedure
    os.system('cp network_joint_bn_kd.py %s' % (output_path)) # bkp of train procedure
    os.system('cp data_loader.py %s' % (output_path))
    network = Build_network(batch_size=train_config["batch_size"], hyperparam_config=hyperparam_config, cost_kwargs=cost_kwargs)
    logging.info("Network built ...")

    trainer = Trainer(net = network,\
                      source_train_list = source_training_list,\
                      target_train_list = target_training_list,\
                      source_val_list = source_validation_list,\
                      target_val_list = target_validation_list,\
                      output_path = output_path,\
                      train_config = train_config,\
                      version_config = version_config)

    # start tensorboard before getting started
    # command2 = "fuser " + version_config["port"] + "/tcp -k"
    # os.system(command2)
    # command1 = "tensorboard --logdir=" + output_path + " --port=" + version_config["port"] + " &"
    # os.system(command1)

    trainer.train_segmenter(restored_model = restored_model)

if __name__ == "__main__":

    restored_model = None#'/research/pheng4/qdliu/Dou_Project/3D-UNet/output/_0308_5/checkpoint/_model.cpkt-5200'
    main(restored_model = restored_model)