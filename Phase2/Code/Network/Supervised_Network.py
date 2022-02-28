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


import torch.nn as nn
import torch


class HomographyNet(nn.Module):
    def __init__(self):
        super(HomographyNet, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(2, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(64, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(128 * 16 * 16, 1024),
                                nn.Dropout(0.5),
                                nn.Linear(1024, 8))

    def forward(self, x):
        out = self.layers(x)
        out = out.contiguous().view(-1, 128 * 16 * 16)

        out = self.fc(out)
        return out

if __name__=='__main__':
    input = torch.randn(1,2,128,128).float()
    model = HomographyNet()
    output = model(input)
    print(output.shape)