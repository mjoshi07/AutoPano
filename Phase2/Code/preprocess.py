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
import random
import numpy as np
import os


def generate_data(root_dir, save_path, txt_file):
    img_paths = open(txt_file).readlines()
    rho = 32
    save_path = os.path.join(root_dir, save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for img_path in img_paths:
        abs_path = os.path.join(root_dir, img_path[:-1]) +'.jpg'
        img_name = abs_path.split("/")[-1]
        img = cv2.imread(abs_path, 0)
        img = cv2.resize(img, (320, 240))
        patch_w, patch_h = 128, 128
        c_a = [[rho, rho],
               [rho+patch_w, rho],
               [rho+patch_w, rho+patch_h],
               [rho, rho+patch_h]]

        c_b = [[rho+random.randint(-rho, rho), rho+random.randint(-rho, rho)],
               [rho+patch_w+random.randint(-rho, rho), rho+random.randint(-rho, rho)],
               [rho+patch_w+random.randint(-rho, rho), rho+patch_h+random.randint(-rho, rho)],
               [rho+random.randint(-rho, rho), rho+patch_h+random.randint(-rho, rho)]]

        H = cv2.getPerspectiveTransform(np.float32(c_a), np.float32(c_b))
        warped_image = cv2.warpPerspective(img, np.linalg.inv(H), (320, 240))

        img_tile = img[c_a[0][1]:c_a[2][1], c_a[0][0]:c_a[2][0]]
        warped_img_tile = warped_image[c_a[0][1]:c_a[2][1], c_a[0][0]:c_a[2][0]]

        training_image = np.dstack((img_tile, warped_img_tile))
        H4pt = np.subtract(np.array(c_b), np.array(c_a))
        data = (training_image, H4pt)
        np.save(os.path.join(save_path, img_name[:-4]), data)

