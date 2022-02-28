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
import random
import pandas as pd
import os


def generate_image_pair(img, patch_size = 128, rho = 32, border_margin = 42):
    h,w = img.shape[:2]
    min_dim = patch_size + 2*border_margin+1
    if w > min_dim and h > min_dim:
        end_margin = patch_size + border_margin

        x = random.randint(border_margin, w-end_margin)
        y = random.randint(border_margin, h-end_margin)

        pts1 = np.array([[x,y], [x, patch_size+y] , [patch_size+x, y], [patch_size+x, patch_size+y]])
        pts2 = np.zeros_like(pts1)

        for i,pt in enumerate(pts1):
            pts2[i][0] = pt[0] + random.randint(-rho, rho)
            pts2[i][1] = pt[1] + random.randint(-rho, rho)

        H = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))

        warp_img = cv2.warpPerspective(img, np.linalg.inv(H), (w,h))

        img_tile = img[y:y+patch_size, x:x+patch_size]
        warp_img_tile = warp_img[y:y+patch_size, x:x+patch_size]
        H4 = (pts2 - pts1).astype(np.float32)

        return img_tile, warp_img_tile, H4, warp_img, np.dstack((pts1,pts2))
    else:
        return None, None, None, None, None

def main():
    all_H4 = []
    all_images = []
    corners = []
    data_gen_modes = ['Test']

    for mode in data_gen_modes:
        noneCounter=0

        if mode == 'Train':
            print("Generating Train data")
            path = '../Data/Train/'
            save_path = '../Data/Train_synthetic/'
            count = 5000

        elif mode == 'Val':
            print("Generating Validation data")
            path = '../Data/Val/'
            save_path = '../Data/Val_synthetic/'
            count = 1000

        else:
            print("Generating Test data")
            path = '../Data/Test/'
            save_path = '../Data/Test_synthetic/'
            count = 1000

        if not os.path.exists(save_path):
            print(save_path, "  was not present, creating the folder...")
            os.makedirs(save_path)

        for i in range(1, count + 1):

            img = cv2.imread(os.path.join(path, str(i)) + '.jpg')
            img = cv2.resize(img, (320,240), interpolation = cv2.INTER_AREA)

            Patch_a, Patch_b, H4, _, points = generate_image_pair(img, patch_size = 128, rho = 32, border_margin = 42)

            if ((Patch_a is None)&(Patch_b is None)&(H4 is None)):
                noneCounter+=1
            else:
                if not os.path.isdir(save_path +'PA/'):
                    os.makedirs(save_path +'PA/')
                    os.makedirs(save_path +'PB/')
                    os.makedirs(save_path +'IA/')

                path_A = save_path +'PA/' + str(i) + 'a.jpg'
                path_B = save_path +'PB/' + str(i) + 'a.jpg'
                orig_img_pathA = save_path +'IA/' + str(i) + 'a.jpg'
                cv2.imwrite(path_A, Patch_a)
                cv2.imwrite(path_B, Patch_b)
                cv2.imwrite(orig_img_pathA, img)
                all_H4.append(np.hstack((H4[:,0] , H4[:,1])))
                corners.append(points)
                all_images.append(str(i) + 'a.jpg')


        print("done")
        print("No. of labels: ", len(all_H4),"No. of images: ", len(all_images), "No. of points: ", np.array(corners).shape,  "No. of patches generated: ",(i-noneCounter))

        df = pd.DataFrame(all_H4)
        df.to_csv(save_path+"H4.csv", index=False)
        print("saved H4 data in:  ", save_path)

        np.save(save_path+"corners.npy", np.array(corners))
        print("saved points data in:  ", save_path)

        df = pd.DataFrame(all_images)
        df.to_csv(save_path+"ImageFileNames.csv", index=False)
        print("saved ImageFiles list  in:  ", save_path)


if __name__ == '__main__':
    main()