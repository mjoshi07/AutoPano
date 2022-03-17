# AutoPano
## Panomoric image stitching

Traditional approach and deep learning approach to estimate homography between 2 sets of images.
Implemented supervised and unsupervised deep learning to estimate homography

## Input Images
<p align="center">
<img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Train/Set1/1.jpg" height=200> <img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Train/Set1/2.jpg" height=200> <img
  src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Train/Set1/3.jpg" height=200>
</p>

## Corner Detection
<p align="center">
<img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Results/Set1/corners_1.png" height=200> <img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Results/Set1/corners_2.png" height=200> <img
  src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Results/Set1/corners_3.png" height=200>
</p>

## Adaptive Non-Maximal Suppression
<p align="center">
<img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Results/Set1/anms_1.png" height=200> <img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Results/Set1/anms_2.png" height=200> <img
  src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Results/Set1/anms_3.png" height=200>
</p>

## Feature Encoding
* We encode each feature point for matching. To obtain encodings, we take a patch of size 41x41 around the feature point, then apply gaussian blur and sub-sample a 8x8 patch from it. This patch is then flattened to a feature vector of size 64x1 and standardized to set mean=0 and variance=1


## Feature Matching
<p align="center">
<img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Results/Set1/matching_1.png" height=200> 
  <img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Results/Set1/matching_2.png" height=200>
</p>

## RANSAC - outlier rejection
<p align="center">
<img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Results/Set1/RANSAC_1.png" height=200> 
  <img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Results/Set1/RANSAC_2.png" height=200>
</p>

## Warping and Blending
<p align="center">
<img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Results/Set1/mypano.png" height=400> 
</p>


## Results on some other data
* Image Stitching using traditional approach

<p align="center">
<img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Results/1.png" height=400><img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase1/Data/Results/2.png" height=400>
</p>

* Homography estimation by deep learning approach
<p align="center">
<img src="https://user-images.githubusercontent.com/31381335/158908397-56f53f11-2ed2-48ac-8980-8c52320b9c73.png" height=300>
</p>

* MCE - Mean Corner Error

<p align="center">
<img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase2/Data/Results/60.png" height=300><img src="https://github.com/mjoshi07/AutoPano/blob/main/Phase2/Data/Results/97.png" height=300>
</p>

