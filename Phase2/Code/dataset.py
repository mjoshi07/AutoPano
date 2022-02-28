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

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob


class HomographyDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.image_pairs = []
        self.homographies = []
        self.data_paths = glob(path+'/*.npy')
        for data_path in self.data_paths:
            np_array = np.load(data_path, allow_pickle=True)
            self.image_pairs.append(torch.from_numpy((np_array[0].astype(float)-127.5)/127.5))
            self.homographies.append(torch.from_numpy(np_array[1].astype(float)/32.0))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        return self.image_pairs[idx], self.homographies[idx]


if __name__=='__main__':
    train_path = "../Data/train_processed"
    trainloader = DataLoader(HomographyDataset(train_path), batch_size=1, shuffle=True)
    train_iter = iter(trainloader)
    img_pair, homography = next(train_iter)
    print(img_pair.shape, homography.shape)