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


import time
import glob
import os
import sys
import numpy as np
import pandas as pd
import cv2

# Don't generate pyc codes
sys.dont_write_bytecode = True


def tic():
    StartTime = time.time()
    return StartTime


def toc(StartTime):
    return time.time() - StartTime


def remap(x, oMin, oMax, iMin, iMax):
    # Taken from https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratios
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if iMin == iMax:
        print("Warning: Zero output range")
        return None

    result = np.add(np.divide(np.multiply(x - iMin, oMax - oMin), iMax - iMin), oMin)

    return result


def FindLatestModel(CheckPointPath):
    FileList = glob.glob(CheckPointPath + '*.ckpt.index') # * means all if need specific format then *.csv
    LatestFile = max(FileList, key=os.path.getctime)
    # Strip everything else except needed information
    LatestFile = LatestFile.replace(CheckPointPath, '')
    LatestFile = LatestFile.replace('.ckpt.index', '')
    return LatestFile


def convertToOneHot(vector, n_labels):
    return np.equal.outer(vector, np.arange(n_labels)).astype(np.float)


"""
auxiliaryMatrices to build the A matrix in Tensor DLT  
referred from :  https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018/blob/master/code/utils/utils.py

""" 

Aux_M1  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)

Aux_M2  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ]], dtype=np.float64)

Aux_M3  = np.array([
          [0],
          [1],
          [0],
          [1],
          [0],
          [1],
          [0],
          [1]], dtype=np.float64)


Aux_M4  = np.array([
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)


Aux_M5  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)

Aux_M6  = np.array([
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ]], dtype=np.float64)

Aux_M71 = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)

Aux_M72 = np.array([
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ]], dtype=np.float64)

Aux_M8  = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ]], dtype=np.float64)
Aux_Mb  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , -1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)


def getPatchIndices(corners_a):
    """
    For a given set of 4 corners, return it's indices inside the region as a mesh grid
    -used in unsupervised model
    """
    
    patch_indices = []
    for i in range(corners_a.shape[0]):
        xmin,ymin = corners_a[i,0,0], corners_a[i,0,1]
        xmax,ymax = corners_a[i,3,0], corners_a[i,3,1]
        X, Y = np.mgrid[xmin:xmax, ymin:ymax]
        patch_indices.append(np.dstack((Y,X))) 
    return np.array(patch_indices)


def draw_corners(image, corners, color):

    corners_array = np.array(corners.copy())
    r = corners_array[2, :].copy()
    corners_array[2, :] = corners_array[3, :]
    corners_array[3, :] = r
    corners_array = corners_array.reshape(-1, 1, 2)
    corners_array = corners_array.astype(int)

    output_image = cv2.polylines(image, [corners_array], True, color, 4)

    return output_image


def compute_homo_from_H4pt(corners1, H4pt):

    H4pt = H4pt.reshape(2, 4).T
    corners_a = np.array(corners1)
    corners_b = corners1 + H4pt
    homo = cv2.getPerspectiveTransform(np.float32(corners_a), np.float32(corners_b))

    return homo


def draw(i, BasePath, SavePath):

    img_patch_names = pd.read_csv(BasePath+"/ImageFileNames.csv")
    img_patch_names = img_patch_names.to_numpy()
    img_a_path = BasePath + '/IA/' + img_patch_names[i, 0]
    img_a = cv2.imread(img_a_path)

    H_pred = np.load(SavePath+"/H_Pred.npy")
    H4pt_true = pd.read_csv(BasePath+"/H4.csv", index_col=False)
    H4pt_true = H4pt_true.to_numpy()

    points = np.load(BasePath+'/corners.npy')
    corners_img_a = points[i, :, :, 0]

    H_true = compute_homo_from_H4pt(corners_img_a, H4pt_true[i])
    corners_a = np.array(corners_img_a)
    corners_a = corners_a.reshape((-1, 1, 2))

    corners_b_true = cv2.perspectiveTransform(np.float32(corners_a), H_true)
    corners_b_true = corners_b_true.astype(int)
    corners_b_pred = cv2.perspectiveTransform(np.float32(corners_a), H_pred[i])
    corners_b_pred = corners_b_pred.astype(int)

    corners_on_img_a = draw_corners(img_a, corners_b_true, (0, 255, 0))
    corners_on_img_a = draw_corners(corners_on_img_a, corners_b_pred, (0, 0, 255))

    mce = np.mean(np.abs(np.array(corners_b_pred) - np.array(corners_b_true)))

    output_img = cv2.putText(corners_on_img_a, "MCE: "+str(round(mce, 3)), (160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 4, cv2.LINE_AA)
    output_img = cv2.putText(output_img, "MCE: "+str(round(mce, 3)), (160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    return output_img
