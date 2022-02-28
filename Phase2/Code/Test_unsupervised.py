"""
CMSC733 Spring 2022: Classical and Deep Learning Approaches for Geometric Computer Vision
Project1: MyAutoPano: Phase 2

Author(s):
Mayank Joshi
Masters student in Robotics,
University of Maryland, College Park

Adithya Gaurav Singh
Masters student in Robotics,
University of Maryland, College Park
"""

from Misc.MiscUtils import *
from Misc.DataUtils import *
from Network.Unsupervised_Network import *
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def load_test_data(folder_name, files_in_dir, points_list, NumTestSamples):

    patch_pairs = []
    corners1 = []
    patches2 = []
    images1 = []

    for n in range(NumTestSamples):

        index = n
        patch1_name = folder_name + os.sep + "PA/" + files_in_dir[index, 0]
        patch1 = cv2.imread(patch1_name, cv2.IMREAD_GRAYSCALE)

        patch2_name = folder_name + os.sep + "PB/" + files_in_dir[index, 0]
        patch2 = cv2.imread(patch2_name, cv2.IMREAD_GRAYSCALE)

        image1_name = folder_name + os.sep + "IA/" + files_in_dir[index, 0]
        image1 = cv2.imread(image1_name, cv2.IMREAD_GRAYSCALE)

        if patch1 is None or patch2 is None:
            continue

        patch1 = np.float32(patch1)
        patch2 = np.float32(patch2)
        image1 = np.float32(image1)

        patch_pair = np.dstack((patch1, patch2))
        corner1 = points_list[index, :, :, 0]

        patch_pairs.append(patch_pair)
        corners1.append(corner1)
        patches2.append(patch2.reshape(128, 128, 1))

        images1.append(image1.reshape(image1.shape[0], image1.shape[1], 1))

    patch_indices = getPatchIndices(np.array(corners1))
    return np.array(patch_pairs), np.array(corners1), np.array(patches2), np.array(images1), patch_indices


def inference(img_pairs_PH, corners_PH, img2_PH, img1_PH,patch_idx_PH, Model_Path, Base_Path, all_files, corners_list, Save_Path, Num_Samples):

    _, H_batches = unsupervised_HomographyNet(img_pairs_PH, corners_PH, img1_PH, patch_idx_PH, Num_Samples)

    Saver = tf.train.Saver()
    with tf.Session() as sess:
        Saver.restore(sess, Model_Path)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(x.get_shape().as_list()) for x in tf.trainable_variables()]))

        img_pairs, corner1, img2, img1, img_idx = load_test_data(Base_Path, all_files, corners_list, Num_Samples)
        feed_dict = {img_pairs_PH: img_pairs, corners_PH: corner1, img2_PH: img2, img1_PH: img1, patch_idx_PH: img_idx}

        H_pred = sess.run(H_batches, feed_dict)
        np.save(os.path.join(Save_Path, 'H_Pred.npy'), H_pred)


def run_unsupervised(ModelPath, BasePath, SavePath, NumTestSamples):

    if not os.path.exists(SavePath):
        print(SavePath)
        os.makedirs(SavePath)

    all_files, SaveCheckPoint, ImageSize, _, _ = SetupAll(BasePath)

    MaxSamplesForTest = 100
    if NumTestSamples > MaxSamplesForTest:
        print("Can Test for only atmax of 100 samples, setting test sample size to 100")
        NumTestSamples = 100

    print("Images for Testing: ",  NumTestSamples)
    corners_list = np.load(BasePath+'/corners.npy')

    corners_PH = tf.placeholder(tf.float32, shape=(MaxSamplesForTest, 4, 2))
    img_pair_PH = tf.placeholder(tf.float32, shape=(MaxSamplesForTest, 128, 128, 2))
    img2_PH = tf.placeholder(tf.float32, shape=(MaxSamplesForTest, 128, 128, 1))
    img1_PH = tf.placeholder(tf.float32, shape=(MaxSamplesForTest, 240, 320, 1))
    patch_idx_PH = tf.placeholder(tf.int32, shape=(MaxSamplesForTest, 128, 128, 2))

    inference(img_pair_PH, corners_PH, img2_PH, img1_PH, patch_idx_PH, ModelPath, BasePath, all_files, corners_list, SavePath, MaxSamplesForTest)

    rand_i = np.random.randint(0, MaxSamplesForTest, size=NumTestSamples)
    for i in rand_i:
        comparison = draw(i, BasePath, SavePath)
        print("Processing image : " + str(i + 1) + 'a.jpg')
        cv2.imwrite(SavePath+'//' + str(i)+'.png', comparison)
    print('Check Results/unsupervised folder..')

