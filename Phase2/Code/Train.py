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

import argparse


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelType', default='supervised', help='Specify model name')

    Args = Parser.parse_args()

    if Args.ModelType=='supervised':
        from Train_supervised import run_supervised_training
        run_supervised_training(n_epochs=50)
    else:
        from Train_unsupervised import run_unsupervised_training
        base_path = "../Data/Train_synthetic"
        checkpoint_path = '../Checkpoints/'
        num_epochs = 50
        batch_size = 16
        logs_path = '../Logs'
        run_unsupervised_training(BasePath=base_path, CheckPointPath=checkpoint_path,
                                NumEpochs=num_epochs,batch_size=batch_size, LogsPath=logs_path)

if __name__=='__main__':
    main()