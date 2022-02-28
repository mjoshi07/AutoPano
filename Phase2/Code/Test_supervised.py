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

import cv2
import numpy as np
import os
from glob import glob
from Network.Supervised_Network import HomographyNet
import random

import torch
from Misc.MiscUtils import draw_corners


def run_supervised(ModelPath, BasePath, SavePath, NumTestSamples):

    BasePath += "/IA/"

    if not os.path.exists(SavePath):
        print(SavePath)
        os.makedirs(SavePath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HomographyNet().to(device)
    rho = 32
    if torch.cuda.is_available():
        ckpt = torch.load(ModelPath)
    else:
        ckpt = torch.load(ModelPath, map_location='cpu')

    model.load_state_dict(ckpt)
    model.eval()

    imgs_paths = glob(os.path.join(BasePath, "*.jpg"))
    random.shuffle(imgs_paths)

    for _ in range(NumTestSamples):
        img_path = random.choice(imgs_paths)
        img_name = os.path.basename(img_path)
        print("Processing image : ", img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (320, 240))
        img_draw = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        patch_w, patch_h = 128, 128
        c_a = [[rho, rho],
               [rho + patch_w, rho],
               [rho + patch_w, rho + patch_h],
               [rho, rho + patch_h]]

        c_b = [[rho + random.randint(-rho, rho), rho + random.randint(-rho, rho)],
               [rho + patch_w + random.randint(-rho, rho), rho + random.randint(-rho, rho)],
               [rho + patch_w + random.randint(-rho, rho), rho + patch_h + random.randint(-rho, rho)],
               [rho + random.randint(-rho, rho), rho + patch_h + random.randint(-rho, rho)]]

        H = cv2.getPerspectiveTransform(np.float32(c_a), np.float32(c_b))
        warped_image = cv2.warpPerspective(img, np.linalg.inv(H), (320, 240))

        img_tile = img[c_a[0][1]:c_a[2][1], c_a[0][0]:c_a[2][0]]
        warped_img_tile = warped_image[c_a[0][1]:c_a[2][1], c_a[0][0]:c_a[2][0]]
        H4pt_gt = np.subtract(np.array(c_b), np.array(c_a)).astype(float) / 32.0
        test_image = np.dstack((img_tile, warped_img_tile))

        input_image = torch.from_numpy((test_image.astype(float) - 127.5) / 127.5).unsqueeze(dim=0)
        input_image = input_image.to(device).permute(0, 3, 1, 2).float()
        H4pt_pred = model(input_image).squeeze(dim=0).cpu().data.numpy().reshape((4, 2))
        gt = H4pt_gt.reshape((1, 8))
        pred = H4pt_pred.reshape((1, 8))
        mse = np.square(np.subtract(gt, pred)).mean()

        H4pt_pred *= 32.0
        c_b_pred = np.add(H4pt_pred, np.array(c_a)).astype('int')

        top_gt, left_gt, bottom_gt, right_gt = c_b[0], c_b[1], c_b[2], c_b[3]
        top_pred, left_pred, bottom_pred, right_pred = c_b_pred[0], c_b_pred[1], c_b_pred[2], c_b_pred[3]

        mce = np.mean(np.abs(np.array(c_b) - np.array(c_b_pred)))  # mean_corner_error

        corners_gt = np.array([[left_gt, top_gt], [right_gt, top_gt], [right_gt, bottom_gt], [left_gt, bottom_gt]])
        corners_pred = np.array(
            [[left_pred, top_pred], [right_pred, top_pred], [right_pred, bottom_pred], [left_pred, bottom_pred]])

        img_draw = draw_corners(img_draw, corners_gt, (0, 255, 0))
        img_draw = draw_corners(img_draw, corners_pred, (0, 0, 255))

        img_draw = cv2.putText(img_draw, "MCE: " + str(round(mce, 3)), (160, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 4, cv2.LINE_AA)
        img_draw = cv2.putText(img_draw, "MCE: " + str(round(mce, 3)), (160, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imwrite(os.path.join(SavePath, img_name), img_draw)

    print("Saved results in Results/supervised")
