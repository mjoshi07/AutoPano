#!/usr/bin/env python

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

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import HomographyDataset
from Network.Supervised_Network import HomographyNet
from tqdm import tqdm
from preprocess import generate_data


def train(model, opt, crit, dataloader, device):
    model.train()
    total_loss = 0.0
    count = 0
    pbar = tqdm(dataloader)
    for image_pair, target in pbar:
        loss = 0.0
        count+=1
        image_pair = image_pair.to(device)
        target = target.to(device)
        image_pair, target = image_pair.permute(0,3,1,2).float(), target.float()
        pred = model(image_pair)
        loss = crit(pred, target.view(-1,8))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss+=loss.item()
        pbar.set_description("Loss {:.3f} | Avg Loss {:.3f}".format(loss.item(), total_loss/count))
    return total_loss / count


def validate(model, crit, dataloader, device):
    model.eval()
    pbar = tqdm(dataloader)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for image_pair, target in pbar:
            loss = 0.0
            count += 1
            image_pair = image_pair.to(device)
            target = target.to(device)
            image_pair, target = image_pair.permute(0, 3, 1, 2).float(), target.float()
            pred = model(image_pair)
            loss = crit(pred, target.view(-1, 8))
            total_loss += loss.item()
            pbar.set_description("Loss {:.3f} | Avg Loss {:.3f}".format(loss.item(), total_loss / count))
    return total_loss / count


def run_supervised_training(n_epochs=50):
    print("Generate synthetic data for training")
    generate_data('../Data', 'train_processed/', 'TxtFiles/DirNamesTrain.txt')
    generate_data('../Data', 'val_processed/', 'TxtFiles/DirNamesVal.txt')
    print("Data Generated")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HomographyNet().to(device)
    train_batch_size = 64
    valid_batch_size = 64
    train_file = "../Data/train_processed"
    valid_file = "../Data/val_processed"
    checkpoint_path = '../Checkpoints/supervised/checkpoint.pth'
    trainloader = DataLoader(HomographyDataset(train_file), batch_size=train_batch_size, shuffle=True)
    valloader = DataLoader(HomographyDataset(valid_file), batch_size=valid_batch_size, shuffle=False)
    print(len(trainloader))
    loss_fn = nn.MSELoss()
    lr = 0.005
    momentum = 0.9
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optim_sched = torch.optim.lr_scheduler.MultiStepLR(optim, np.arange(0, n_epochs, 5), gamma=0.1)
    val_every = 5
    training_losses = []
    val_losses = []
    val_losses.append(validate(model, loss_fn, valloader, device))
    for i in range(n_epochs):
        training_losses.append(train(model, optim, loss_fn, trainloader, device))
        optim_sched.step(i)

        if i % val_every==0 and i!=0:
            val_losses.append(validate(model, loss_fn, valloader, device))
        torch.save(model.state_dict(), checkpoint_path)
