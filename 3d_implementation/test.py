import os
import sys
import logging
import time
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from network import Build_network
from data_loader import parse_fn, _label_decomp
import datetime
import glob
# import matplotlib.pyplot as plt


if not str(sys.argv[1]):
    print('No GPU given... setting to 0')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1])


test_batch_size =1

class_map = {  # a map used for mapping label value to its name, used for output
    "0": "bg",
    "1": "liver",
    "2": "right kidney",
    "3": "left kidney",
    "4": "spleen" }

hyperparam_config = {
    "patch_size": [256,256,8],
    "input_channels": 1,
    "num_class": 5,
    "feature_base": 16,
}
cost_kwargs = {
    "miu_seg_dice": 1.0,
    "miu_seg_ce": 1.0,
    "miu_seg_L2_norm": 1e-4,
    "miu_kd": 0
}

version_config = {
    "prefix": '',
    "date": 'sifn_0.5',
}
logging.basicConfig(filename='./output/log/' + version_config["date"] + '_test_log', level=logging.DEBUG, format='%(asctime)s %(message)s')
currtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
logging.getLogger().addHandler(logging.StreamHandler())

class Tester(object):

    def __init__(self, net, source_test_list, target_test_list, output_path):

        self.net = net
        self.source_test_list = source_test_list
        self.target_test_list = target_test_list
        self.output_path = output_path

        with open(self.source_test_list, 'r') as fp:
            rows = fp.readlines()
        self.source_image_list = [row[:-1] if row[-1] == '\n' else row for row in rows]

        with open(self.target_test_list, 'r') as fp:
            rows = fp.readlines()
        self.target_image_list = [row[:-1] if row[-1] == '\n' else row for row in rows]

        # self.image_list = self.image_list[0:5]

    def patch_process(self, filename, split=False, stitch=False, pred_patch=None):
        # extract image into patches, and build into a batch
        # checked location correspondence for this function
        image, mask = parse_fn(filename)

        sizeX, sizeY, sizeZ = image.shape
        patchX, patchY, patchZ = self.net.patch_size
        numX, numY, numZ = [int(np.ceil(1.0 * sizeX / patchX)), int(np.ceil(1.0 * sizeY / patchY)), int(np.ceil(1.0 * sizeZ / patchZ))]
        padX, padY, padZ = [patchX * numX - sizeX, patchY * numY - sizeY, patchZ * numZ - sizeZ]

        if split is True:

            patch_wise_image = []
            patch_wise_mask = []
            patch_num = 0

            for k in xrange(numZ):
                if k != (numZ - 1):
                    a = image[:, :, k*patchZ:(k+1)*patchZ]
                    patch_wise_image.append(a)
                    b = mask[:, :, k*patchZ:(k+1)*patchZ]
                    patch_wise_mask.append(b)
                else:
                    a = image[:, :, sizeZ-patchZ:]
                    patch_wise_image.append(a)
                    b = mask[:, :, sizeZ-patchZ:]
                    patch_wise_mask.append(b)
            patch_wise_image = np.expand_dims(np.stack(patch_wise_image), -1)
            patch_wise_mask  = _label_decomp(np.stack(patch_wise_mask), num_cls=self.net.num_class)

            #print 'per-case testing patch_wise_image size: ', patch_wise_image.shape
            #print 'per-case testing patch_wise_mask size: ', patch_wise_mask.shape

            return image, mask, patch_wise_image, patch_wise_mask

        if stitch is True:

            whole_mask = np.zeros(shape=(patchX*numX, patchY*numY, patchZ*numZ))
            patch_num = 0
            for k in xrange(numZ):
                for j in xrange(numY):
                    for i in range(numX):
                        whole_mask[i*patchX:(i+1)*patchX, j*patchY:(j+1)*patchY, k*patchZ:(k+1)*patchZ] = pred_patch[patch_num,:,:,:]
                        patch_num += 1
            real_mask = whole_mask[leftX:patchX*numX-rightX, leftY:patchY*numY-rightY, leftZ:patchZ*numZ-rightZ]

            return real_mask

    def concat_mask(self, mask, gt):

        sizeX, sizeY, sizeZ = gt.shape
        patchX, patchY, patchZ = self.net.patch_size
        numX, numY, numZ = [int(np.ceil(1.0 * sizeX / patchX)), int(np.ceil(1.0 * sizeY / patchY)), int(np.ceil(1.0 * sizeZ / patchZ))]

        real_mask = np.zeros(gt.shape)
        for k in xrange(numZ):
            if k != (numZ - 1):
                real_mask[:, :, k*patchZ:(k+1)*patchZ] = mask[k, :, :, :]
            else:
                real_mask[:, :, sizeZ-patchZ:] = mask[k, :, :, :]
        return real_mask

    def eva_dice(self, pred, gt, detail=False):
        '''
        :param pred: whole brain prediction
        :param gt: whole
        :param detail:
        :return: a list, indicating Dice of each class for one case
        '''
        dice = []

        for cls in range(self.net.num_class):
            pred_i = np.zeros(pred.shape)
            pred_i[pred == cls] = 1
            gt_i = np.zeros(gt.shape)
            gt_i[gt == cls] = 1
            dice_cls = 2.0 * np.sum(pred_i * gt_i) / (np.sum(pred_i) + np.sum(gt_i))
            dice.append(dice_cls)

            if detail is True:
                logging.info("class {}, dice is {:4f}".format(class_map[str(cls)], dice_cls))
        logging.info("4 class average dice is {:4f}".format(np.mean(dice)))

        return dice


    def save_nii(self, filename, raw_array, seg_mask=False, recon_image=False):

        if seg_mask is True:
            if not os.path.exists(os.path.join(self.output_path, 'pred_seg_masks')):
                logging.info("Allocating '{:}'".format(os.path.join(self.output_path, 'pred_seg_masks')))
                os.makedirs(os.path.join(self.output_path, 'pred_seg_masks'))

            # sitk_image = sitk.ReadImage(os.path.join(filename, 'T1_brain_seg_rigid_to_mni.nii.gz'))
            # saved_nii = sitk.GetImageFromArray(raw_array.transpose([2,0,1]))
            # saved_nii.SetSpacing(sitk_image.GetSpacing())
            # saved_nii.SetOrigin(sitk_image.GetOrigin())
            # saved_nii.SetDirection(sitk_image.GetDirection())
            saved_nii = sitk.GetImageFromArray(raw_array.transpose([2,0,1]))
            saved_name = filename.split('/')[-1] 
            sitk.WriteImage(saved_nii, os.path.join(self.output_path, 'pred_seg_masks', saved_name))

        elif recon_image is True:

            if not os.path.exists(os.path.join(self.output_path, 'recon_images')):
                logging.info("Allocating '{:}'".format(os.path.join(self.output_path, 'recon_images')))
                os.makedirs(os.path.join(self.output_path, 'recon_images'))

            sitk_image = sitk.ReadImage(os.path.join(filename, 'T2_FLAIR_unbiased_brain_rigid_to_mni.nii.gz'))
            saved_nii = sitk.GetImageFromArray(raw_array.transpose([2,0,1]))
            saved_nii.SetSpacing(sitk_image.GetSpacing())
            saved_nii.SetOrigin(sitk_image.GetOrigin())
            saved_nii.SetDirection(sitk_image.GetDirection())

            saved_name = filename.split('/')[-1] + '_recon.nii.gz'
            sitk.WriteImage(saved_nii, os.path.join(self.output_path, 'recon_images', saved_name))

        else:
            logging.info('Please specify what to save.')


    def test_segmenter(self, test_model):

        # init_glb = tf.global_variables_initializer()
        # init_loc = tf.variables_initializer(tf.local_variables())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            saver = tf.train.Saver()
            saver.restore(sess, test_model)
            logging.info("test for model %s" % test_model)

            dice_all = []
            source_prop_list = []
            target_prop_list = []
            for idx_file, filename in enumerate(self.source_image_list):
                start_time = time.time()

                pred_mask_patches = None
                raw_img, gt_mask, test_x, test_y = self.patch_process(filename=filename, split=True)
                all_batch_size = test_x.shape[0]

                for i in xrange(all_batch_size // test_batch_size):
                    source_prob, pred_mask_patch_this = sess.run([self.net.source_prob, self.net.source_pred_compact], feed_dict={self.net.source: test_x[i*test_batch_size:(i+1)*test_batch_size],
                                                                                     self.net.source_y: test_y[i*test_batch_size:(i+1)*test_batch_size],
                                                                                     self.net.target: test_x[i*test_batch_size:(i+1)*test_batch_size],
                                                                                     self.net.target_y: test_y[i*test_batch_size:(i+1)*test_batch_size],
                                                                                     self.net.training_mode: False,
                                                                                     self.net.keep_prob: 1,
                                                                                    })
                    source_prop_list.append(source_prob)
                    if i == 0:
                        pred_mask_patches = pred_mask_patch_this
                    else:
                        pred_mask_patches = np.concatenate((pred_mask_patches, pred_mask_patch_this), axis=0)
                whole_pred_mask = self.concat_mask(pred_mask_patches, gt_mask)
                dice = self.eva_dice(whole_pred_mask, gt_mask)
                dice_all.append(dice)
                self.save_nii(filename, whole_pred_mask, seg_mask=True)

            dice_class = zip(*dice_all)
            logging.info("____________________source average_____________________")
            for cls in xrange(self.net.num_class):
                logging.info("class {}, Dice average is {:4f}, std is {:4f}".format(class_map[str(cls)], np.mean(dice_class[cls]), np.std(dice_class[cls])))
            source_average = (np.mean(dice_class[1]) + np.mean(dice_class[2]) + np.mean(dice_class[3]) +np.mean(dice_class[4])) /4.0
            logging.info("4 class average is {:4f}".format(source_average))

            # test target ---
            dice_all = []
            for idx_file, filename in enumerate(self.target_image_list):
                ## process one case each time
                #logging.info("--------processing # %s case: %s" % (str(idx_file), filename.split("/")[-1]))
                start_time = time.time()

                pred_mask_patches = None
                raw_img, gt_mask, test_x, test_y = self.patch_process(filename=filename, split=True)
                all_batch_size = test_x.shape[0]
                #print test_x.shape
                for i in xrange(all_batch_size // test_batch_size):
                    target_prob, pred_mask_patch_this = sess.run([self.net.target_prob, self.net.target_pred_compact], feed_dict={self.net.target: test_x[i*test_batch_size:(i+1)*test_batch_size],
                                                                                     self.net.target_y: test_y[i*test_batch_size:(i+1)*test_batch_size],
                                                                                     self.net.source: test_x[i*test_batch_size:(i+1)*test_batch_size],
                                                                                     self.net.source_y: test_y[i*test_batch_size:(i+1)*test_batch_size],
                                                                                     self.net.training_mode: False,
                                                                                     self.net.keep_prob: 1,
                                                                                    })
                    print i
                    target_prop_list.append(target_prob)
                    if i == 0:
                        pred_mask_patches = pred_mask_patch_this
                    else:
                        pred_mask_patches = np.concatenate((pred_mask_patches, pred_mask_patch_this), axis=0)
                whole_pred_mask = self.concat_mask(pred_mask_patches, gt_mask)
                dice = self.eva_dice(whole_pred_mask, gt_mask)
                dice_all.append(dice)
                self.save_nii(filename, whole_pred_mask, seg_mask=True)

            dice_class = zip(*dice_all)
            logging.info("____________________target average_____________________")
            for cls in xrange(self.net.num_class):
                logging.info("class {}, Dice average is {:4f}, std is {:4f}".format(class_map[str(cls)], np.mean(dice_class[cls]), np.std(dice_class[cls])))
            target_average = (np.mean(dice_class[1]) + np.mean(dice_class[2]) + np.mean(dice_class[3]) +np.mean(dice_class[4])) /4.0
            logging.info("4 class average is {:4f}".format(target_average))

            logging.info("____________________two modality average_____________________")
            logging.info("2 modality average is {:4f}".format((source_average + target_average)/2.0))

            source_probability = np.array(source_prop_list)
            target_probability = np.array(target_prop_list)
            
            if not os.path.exists(os.path.join(self.output_path, 'array')):
                os.makedirs(os.path.join(self.output_path, 'array'))

            np.save(self.output_path + '/array/' + test_model[-5:] + '_source_array', source_probability)
            np.save(self.output_path + '/array/' + test_model[-5:] + '_target_array', target_probability)

def main(test_model=None, version_config=None):

    # base_path = "/data/projects/test_specific"
    base_path = './'
    source_test_list = base_path + "/data/mr_test_list"
    target_test_list = base_path + "/data/ben_ct_test_list"

    output_path = base_path + "/output/" + version_config["prefix"] + "_" + version_config["date"]
    if not os.path.exists(output_path):
        logging.info("Allocating '{:}'".format(output_path))
        os.makedirs(output_path)

    network = Build_network(batch_size=test_batch_size, hyperparam_config=hyperparam_config, cost_kwargs=cost_kwargs)
    logging.info("Network built ...")
    tester = Tester(net = network, source_test_list = source_test_list, target_test_list = target_test_list, output_path = output_path)

    test_model = "xxxx"
    tester.test_segmenter(test_model = test_model)

if __name__ == "__main__":
        main(version_config=version_config)