#!/usr/bin/evn python

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

# Code starts here:

# Add any python libraries here
import argparse
from Test_unsupervised import run_unsupervised
from Test_supervised import run_supervised
import os


def main():
	# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ModelType', default='supervised', help='Specify model name')

	Args = Parser.parse_args()
	NumTestSamples = 5
	"""
	Read a set of images for Panorama stitching
	"""

	if not os.path.exists(base_path):
		print("[ERROR]: Test Path does not exists")

	"""
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""
	if Args.ModelType == 'supervised':
		base_path = "../Data/Test/"
		model_path = "../Checkpoints/supervised/checkpoint_homography.pth"
		save_path = "Results/supervised/"
		run_supervised(ModelPath=model_path, BasePath=base_path, SavePath=save_path, NumTestSamples=NumTestSamples)
	else:
		base_path = "../Data/Test_synthetic/"
		model_path = "../Checkpoints/unsupervised/99model.ckpt"
		save_path = "Results/unsupervised/"
		run_unsupervised(ModelPath=model_path, BasePath=base_path, SavePath=save_path, NumTestSamples=NumTestSamples)


if __name__ == '__main__':
	main()
 
