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

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import numpy as np
from Misc.MiscUtils import *
from Misc.TFSpatialTransformer import transformer

# Don't generate pyc codes
sys.dont_write_bytecode = True


def TensorDLT(H4_pt, corners_img_1, batch_size):

    """
    TensorDLT code referred from
    https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018/blob/master/code/homography_model.py
    """

    corners_a_tile = tf.expand_dims(corners_img_1, [2]) # batch_size x 8 x 1

    # Solve for H using DLT
    pred_h4p_tile = tf.expand_dims(H4_pt, [2]) # batch_size x 8 x 1
    pred_points_tile = tf.add(pred_h4p_tile, corners_a_tile)

    # obtain 8 auxiliary tensors -> expand dimensions by 1 at first,-> create batch_size number of copies
    tensor_M1 = tf.constant(Aux_M1,tf.float32)
    tensor_M1 =  tf.expand_dims(tensor_M1 ,[0])
    M1_tile = tf.tile(tensor_M1,[batch_size,1,1])

    tensor_M2 = tf.constant(Aux_M2,tf.float32)
    tensor_M2 = tf.expand_dims(tensor_M2,[0])
    M2_tile = tf.tile(tensor_M2,[batch_size,1,1])

    tensor_M3 = tf.constant(Aux_M3,tf.float32)
    tensor_M3 = tf.expand_dims(tensor_M3,[0])
    M3_tile = tf.tile(tensor_M3,[batch_size,1,1])

    tensor_M4 = tf.constant(Aux_M4,tf.float32)
    tensor_M4 = tf.expand_dims(tensor_M4,[0])
    M4_tile = tf.tile(tensor_M4,[batch_size,1,1])

    tensor_M5 = tf.constant(Aux_M5,tf.float32)
    tensor_M5 = tf.expand_dims(tensor_M5,[0])
    M5_tile = tf.tile(tensor_M5,[batch_size,1,1])

    tensor_M6 = tf.constant(Aux_M6,tf.float32)
    tensor_M6 = tf.expand_dims(tensor_M6,[0])
    M6_tile = tf.tile(tensor_M6,[batch_size,1,1])

    tensor_M71 = tf.constant(Aux_M71,tf.float32)
    tensor_M71 = tf.expand_dims(tensor_M71,[0])
    M71_tile = tf.tile(tensor_M71,[batch_size,1,1])

    tensor_M72 = tf.constant(Aux_M72,tf.float32)
    tensor_M72 = tf.expand_dims(tensor_M72,[0])
    M72_tile = tf.tile(tensor_M72,[batch_size,1,1])

    tensor_M8 = tf.constant(Aux_M8,tf.float32)
    tensor_M8 = tf.expand_dims(tensor_M8,[0])
    M8_tile = tf.tile(tensor_M8,[batch_size,1,1])

    tensor_Mb = tf.constant(Aux_Mb,tf.float32)
    tensor_Mb = tf.expand_dims(tensor_Mb,[0])
    Mb_tile = tf.tile(tensor_Mb,[batch_size,1,1])

    # Form the equations Ax = b to compute H
    # Build A matrix
    A1 = tf.matmul(M1_tile, corners_a_tile)  # Column 1
    A2 = tf.matmul(M2_tile, corners_a_tile)  # Column 2
    A3 = M3_tile # Column 3
    A4 = tf.matmul(M4_tile, corners_a_tile)  # Column 4
    A5 = tf.matmul(M5_tile, corners_a_tile)  # Column 5
    A6 = M6_tile  # Column 6
    A7 = tf.matmul(M71_tile, pred_points_tile) *  tf.matmul(M72_tile, corners_a_tile) # Column 7
    A8 = tf.matmul(M71_tile, pred_points_tile) *  tf.matmul(M8_tile, corners_a_tile)  # Column 8

    # reshape A matrices into 8x1 and stack them column wise
    A = tf.stack([tf.reshape(A1, [-1, 8]), tf.reshape(A2, [-1, 8]), tf.reshape(A3, [-1, 8]), tf.reshape(A4, [-1, 8]),
                 tf.reshape(A5, [-1, 8]), tf.reshape(A6, [-1, 8]), tf.reshape(A7, [-1, 8]), tf.reshape(A8, [-1, 8])], axis=1)
    A = tf.transpose(A, perm=[0, 2, 1])

    # Build b matrix
    b_mat = tf.matmul(Mb_tile, pred_points_tile)

    # Solve  Ax = b
    H_8e1 = tf.matrix_solve(A, b_mat)

    H_33 = tf.ones([batch_size, 1, 1])
    H_9 = tf.concat([H_8e1, H_33], 1)

    H_flat = tf.reshape(H_9, [-1, 9])
    Homo = tf.reshape(H_flat, [-1, 3, 3])

    return Homo


def homographyNet(Img):

    x = tf.layers.conv2d(inputs=Img, name='C1', padding='same', filters=64, kernel_size=[3, 3])
    x = tf.layers.batch_normalization(x, name='BN1')
    x = tf.nn.relu(x, name='R1')

    x = tf.layers.conv2d(inputs=x, name='C2', padding='same', filters=64, kernel_size=[3, 3])
    x = tf.layers.batch_normalization(x, name='BN2')
    x = tf.nn.relu(x, name='R2')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

    x = tf.layers.conv2d(inputs=x, name='C3', padding='same', filters=64, kernel_size=[3, 3])
    x = tf.layers.batch_normalization(x, name='BN3')
    x = tf.nn.relu(x, name='R3')

    x = tf.layers.conv2d(inputs=x, name='C4', padding='same', filters=64, kernel_size=[3, 3])
    x = tf.layers.batch_normalization(x, name='BN4')
    x = tf.nn.relu(x, name='R4')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

    x = tf.layers.conv2d(inputs=x, name='C5', padding='same', filters=128, kernel_size=[3, 3])
    x = tf.layers.batch_normalization(x, name='BN5')
    x = tf.nn.relu(x, name='R5')

    x = tf.layers.conv2d(inputs=x, name='C6', padding='same', filters=128, kernel_size=[3, 3])
    x = tf.layers.batch_normalization(x, name='BN6')
    x = tf.nn.relu(x, name='R6')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

    x = tf.layers.conv2d(inputs=x, name='C7', padding='same', filters=128, kernel_size=[3, 3])
    x = tf.layers.batch_normalization(x, name='BN7')
    x = tf.nn.relu(x, name='R7')

    x = tf.layers.conv2d(inputs=x, name='C8', padding='same', filters=128, kernel_size=[3, 3])
    x = tf.layers.batch_normalization(x, name='BN8')
    x = tf.nn.relu(x, name='R8')

    x = tf.layers.flatten(x)

    x = tf.layers.dense(inputs=x, name='FC1', units=1024, activation=tf.nn.relu)
    x = tf.layers.dropout(x, rate=0.5, training=True, name='D1')
    x = tf.layers.batch_normalization(x, name='BN9')
    H4_pt = tf.layers.dense(inputs=x, name='FC_Final', units=8, activation=None)

    return H4_pt


def unsupervised_HomographyNet(patch_batches, corners_img_1, imgs_1_ph, patch_idx_ph, batch_size=64):

    batch_size, h, w, channels = imgs_1_ph.get_shape().as_list()

    H4_pt_batches = homographyNet(patch_batches)
    corners_img_1 = tf.reshape(corners_img_1, [batch_size, 8])
    Homography_batches = TensorDLT(H4_pt_batches, corners_img_1, batch_size)

    Mat = np.array([[w/2.0, 0., w/2.0],
                  [0., h/2.0, h/2.0],
                  [0., 0., 1.]]).astype(np.float32)

    tensor_M = tf.constant(Mat, tf.float32)
    tensor_M = tf.expand_dims(tensor_M, [0])
    M_batches = tf.tile(tensor_M, [batch_size, 1, 1])

    M_inv = np.linalg.inv(Mat)
    tensor_M_inv = tf.constant(M_inv, tf.float32)
    tensor_M_inv = tf.expand_dims(tensor_M_inv, [0])
    M_inv_batches = tf.tile(tensor_M_inv, [batch_size, 1, 1])

    Homography_scaled = tf.matmul(tf.matmul(M_inv_batches, Homography_batches), M_batches)
    img1_warped, _ = transformer(imgs_1_ph, Homography_scaled, (h, w))

    img1_warped = tf.reshape(img1_warped, [batch_size, h, w])
    patch1_warped = tf.gather_nd(img1_warped, patch_idx_ph, name=None, batch_dims=1)

    patch1_warped = tf.transpose(patch1_warped, perm=[0, 2, 1])
    patch1_warped = tf.reshape(patch1_warped, [batch_size, 128, 128, 1])

    return patch1_warped, Homography_batches
