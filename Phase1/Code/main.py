#!/usr/bin/evn python

"""
CMSC733 Spring 2022: Classical and Deep Learning Approaches for Geometric Computer Vision
Project1: MyAutoPano: Phase 1

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
import numpy as np
import cv2
import os
import argparse

corner_img_count = 0
anms_img_count = 0
enc_img_count = 0
fm_img_count = 0
rns_img_count = 0
EPSILON = 1e-10
curr_out_dir = None


def show_corners(img, corners, display_img=True, save_img=False, header="Corners_"):
    global corner_img_count, anms_img_count, curr_out_dir

    if display_img or save_img:
        img_copy = img.copy()

        for feature_point in corners:
            x, y = feature_point[0], feature_point[1]
            x = int(x.item())
            y = int(y.item())
            cv2.circle(img_copy, (x, y), 1, (0, 0, 255), -1)

        if header == "corners_":
            corner_img_count += 1
            img_count = corner_img_count
        elif header == "anms_":
            anms_img_count += 1
            img_count = anms_img_count

        if display_img:
            print("[INFO]: Displaying Image: {}".format(header + str(img_count) + ".png"))
            cv2.namedWindow(header + str(img_count), cv2.WINDOW_FREERATIO)
            cv2.imshow(header + str(img_count), img_copy)
            cv2.waitKey(0)
        if save_img:
            out_dir = curr_out_dir
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            cv2.imwrite(os.path.join(out_dir, header + str(img_count) + ".png"), img_copy)
            print("[INFO]: Saved Image: {}".format(header + str(img_count) + ".png"))


def detect_corners(img, feature_type=1, num_features=1000, display_img=True, save_img=False):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if feature_type != 1:
        print('\n')
        print("==============================================")
        print("[INFO]: Using Harris corner detection")
        corner_mask = cv2.cornerHarris(gray_img, 2, 3, 0.001)
        corner_mask[corner_mask < 0.001 * corner_mask.max()] = 0
        good_corners = np.where(corner_mask >= 0.001 * corner_mask.max())
        final_corners = []
        for corner_x, corner_y in zip(good_corners[1], good_corners[0]):
            final_corners.append([corner_x, corner_y])
        final_corners = np.array(final_corners)
    else:
        print('\n')
        print("==============================================")
        print("[INFO]: Using Shi-Tomasi corner detection")
        dst = cv2.goodFeaturesToTrack(gray_img, num_features, 0.01, 10)
        final_corners = np.int0(dst)
        final_corners = final_corners.reshape(final_corners.shape[0], 2)
        corner_mask = None

    print("[INFO]: Corners found in Image {}".format(len(final_corners)))
    show_corners(img, final_corners, display_img=display_img, save_img=save_img, header="corners_")

    return final_corners, corner_mask


def compute_ANMS(img, corners, corner_image, n_best=1000, display_img=True, save_img=False):
    print('\n')
    print("==============================================")
    if corner_image is not None:
        print("[INFO]: Started Adaptive Non-Maximal Suppression")
        r = []
        for corner_x_i, corner_y_i in corners:
            min_ED = 10000000000
            for corner_x_j, corner_y_j in corners:
                if corner_image[corner_y_j, corner_x_j] > corner_image[corner_y_i, corner_x_i]:
                    ED = (corner_y_j - corner_y_i) ** 2 + (corner_x_j - corner_x_i) ** 2
                    if ED < min_ED:
                        min_ED = ED
            r.append(min_ED)
        r = np.array(r)
        inds = r.argsort()[::-1]

        corners = corners[inds]
        corners = corners[:n_best]
        print("[INFO]: Corners found After ANMS {}".format(len(corners)))
    else:
        print("[INFO]: Not performing ANMS explicitly as Shi-Tomasi has inbuilt ANMS")

    show_corners(img, corners, display_img=display_img, save_img=save_img, header="anms_")

    return corners


def encode_features(feature_points, src, patch_size=(41, 41), display_img=True, save_img=False):
    """
    feature_points: list of keypoint/feature point around which we will take a patch of size 41x41
    src: input image
    patch_size: a small img which will be encoded as feature vector

    consider only those feature points around which 41x41 patch can be computed, ignore the boundary features
    """

    global enc_img_count, curr_out_dir

    enc_img_count += 1

    # assert patch size to be an odd number
    assert patch_size[0] % 2 != 0
    assert patch_size[1] % 2 != 0

    feature_dict = {}

    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    print('\n')
    print("==============================================")
    print("[INFO]: Started Feature Encoding")
    feature_count = 0
    for feature_point in feature_points:
        feature_count += 1

        x, y = feature_point[0], feature_point[1]
        x = int(x.item())
        y = int(y.item())

        horizontal_half = int((patch_size[0] - 1) / 2)
        vertical_half = int((patch_size[1] - 1) / 2)

        # consider a patch for processing iff it satisfies all of the four condition
        if x - horizontal_half < 0:
            continue
        if y - vertical_half < 0:
            continue
        if x + horizontal_half > src.shape[1]:
            continue
        if y + vertical_half > src.shape[0]:
            continue

        # if all conditions were met then we can take a patch around the feature point: (x, y)
        img_patch = src[y - vertical_half: y + vertical_half, x - horizontal_half: x + horizontal_half]

        kernel_size = (3, 3)
        blurred_patch = cv2.GaussianBlur(img_patch, kernel_size, 0)

        # sub-sample the patch to 8x8
        sub_sample = cv2.resize(blurred_patch, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)

        if display_img:
            cv2.namedWindow("FD_" + str(feature_count), cv2.WINDOW_FREERATIO)
            cv2.imshow("FD_" + str(feature_count), sub_sample)
            cv2.waitKey(0)
        if save_img:
            out_dir = os.path.join(curr_out_dir, str(enc_img_count))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            cv2.imwrite(os.path.join(out_dir, "FD_" + str(feature_count) + ".png"), sub_sample)
            print("[INFO]: Saved Image: {}".format("FD_" + str(feature_count) + ".png"))

        feature_vector = sub_sample.reshape(-1)

        mean = feature_vector.mean()
        std = feature_vector.std()

        # subtract mean
        feature_vector = feature_vector - mean

        # divide by standard deviation
        feature_vector = feature_vector / std

        feature_dict[str(x) + "_" + str(y)] = feature_vector

    print("[INFO]: Completed Feature Encoding")
    print("[INFO]: Number of Feature Encoded are {}".format(len(feature_dict)))

    return feature_dict


def match_features(feature_dict_1, feature_dict_2, min_match_threshold):
    """
    feature_dict_1: corresponds to img1, key: (location of point as x_y) value: (64x1 feature vector)
    feature_dict_2: corresponds to img2, key: (location of point as x_y) value: (64x1 feature vector)
    """
    print('\n')
    print("==============================================")
    print("[INFO]: Started Feature Matching")
    matched_pairs = []
    for key1, val1 in feature_dict_1.items():
        ssd = {}
        for key2, val2 in feature_dict_2.items():
            ssd[key2] = np.linalg.norm(val1 - val2)

        sorted_dict = {k: v for k, v in sorted(ssd.items(), key=lambda item: item[1])}

        k1, score1 = list(sorted_dict.keys())[0], list(sorted_dict.values())[0]
        k2, score2 = list(sorted_dict.keys())[1], list(sorted_dict.values())[1]

        match_ratio = float(score1) / float(score2 + EPSILON)

        if match_ratio <= min_match_threshold:
            # keep this pair of matched point
            # extract points from key1 and key2 and store them
            img1_point = [int(key1.split("_")[0]), int(key1.split("_")[1])]
            img2_point = [int(k1.split("_")[0]), int(k1.split("_")[1])]

            matched_pairs.append([img1_point, img2_point])

    matched_pairs = np.array(matched_pairs)
    print("[INFO]: Completed Feature Matching")
    print("[INFO]: Matches found are {}".format(len(matched_pairs[:, 0])))

    return matched_pairs


def show_matching(img1, img2, points, display_img=True, save_img=False, type=None):
    global fm_img_count, curr_out_dir, rns_img_count

    points = points.copy()
    img_count = None
    header = "matching_"
    if type is None:
        fm_img_count += 1
        img_count = fm_img_count
        points1 = points[:, 0]
        points2 = points[:, 1]
    elif type == "RANSAC":
        rns_img_count += 1
        header = "RANSAC_"
        img_count = rns_img_count
        points1 = points[0]
        points2 = points[1]

    img1 = img1.copy()
    img2 = img2.copy()

    width = img1.shape[1] + img2.shape[1]
    height = max(img1.shape[0], img2.shape[0])
    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    output_img[0: img1.shape[0], 0:img1.shape[1]] = img1
    output_img[0:img2.shape[0], img1.shape[1]:] = img2

    for idx in range(len(points1)):
        point1 = points1[idx]
        point2 = points2[idx]

        # add width of img1 to every point of img2
        point2[0] = point2[0] + img1.shape[1]

        cv2.line(output_img, point1, point2, (0, 255, 255), 1, 16)
        cv2.circle(output_img, point1, 3, (255, 0, 0), 1)
        cv2.circle(output_img, point2, 3, (255, 0, 0), 1)

    if display_img:
        cv2.namedWindow("Feature matching", cv2.WINDOW_FREERATIO)
        cv2.imshow("Feature matching", output_img)
        cv2.waitKey(0)

    if save_img:
        out_path = curr_out_dir
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cv2.imwrite(os.path.join(out_path, header + str(img_count) + ".png"), output_img)
        print("[INFO]: Saved Image: {}".format(header + str(img_count) + ".png"))


def get_inliers(val1, val2, thresh):
    num_vals = val1.shape[0]
    error = np.zeros(num_vals)

    for i in range(num_vals):
        error[i] = np.linalg.norm(val1[i] - val2[i])

    error[error <= thresh] = 1
    error[error > thresh] = 0

    inlier_count = np.sum(error)

    return inlier_count, error


def compute_RANSAC(matched_pairs, iterations=5000, thresh=5.0):

    print('\n')
    print("==============================================")
    print("[INFO]: Started RANSAC for outlier rejection")

    img1_points = matched_pairs[:, 0]
    img2_points = matched_pairs[:, 1]

    max_inlier_count = 0
    best_homo = np.ones([3, 3], np.float32)

    inlier_pair_indices = []
    num_points = img1_points.shape[0]

    for i in range(iterations):

        indices = np.random.choice(num_points, size=4)

        img1_random_points = img1_points[indices]
        img2_random_points = img2_points[indices]

        homo = cv2.getPerspectiveTransform(np.float32(img1_random_points), np.float32(img2_random_points))

        img1_points_stacked = np.vstack((img1_points[:, 0], img1_points[:, 1], np.ones([1, num_points])))
        img1_points_on_img2 = np.matmul(homo, img1_points_stacked)

        points_x = img1_points_on_img2[0, :] / (img1_points_on_img2[2, :] + EPSILON)
        points_y = img1_points_on_img2[1, :] / (img1_points_on_img2[2, :] + EPSILON)

        pred_points = np.array([points_x, points_y]).T
        true_points = img2_points

        inlier_count, error = get_inliers(true_points, pred_points, thresh)

        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_homo = homo
            inlier_pair_indices = np.where(error == 1)

    strong_pairs = np.array([img1_points[inlier_pair_indices], img2_points[inlier_pair_indices]])

    print("[INFO]: Completed RANSAC for outlier rejection")
    print("[INFO]: Strong pairs found are {}".format(len(strong_pairs[0])))

    return strong_pairs, best_homo


def stitch_images(img1, img2, homography):

    img1 = img1.copy()
    img2 = img2.copy()

    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape

    img1_points = np.array([[[0, 0]], [[0, h1]], [[w1, h1]], [[w1, 0]]], dtype=np.float32)
    img2_points = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32)

    img1_points_on_img2 = cv2.perspectiveTransform(img1_points, homography)

    img1_points_on_img2 = img1_points_on_img2.reshape(img1_points_on_img2.shape[0], 2)
    shape = (img1_points_on_img2.shape[0] + img2_points.shape[0], img2_points.shape[1])
    merged_points = np.array([img1_points_on_img2, img2_points]).reshape(shape[0], shape[1])

    x_min, y_min = np.int0(np.min(merged_points, axis=0))
    x_max, y_max = np.int0(np.max(merged_points, axis=0))
    warped_img_shape = (x_max - x_min, y_max - y_min)

    homo_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    warped_img = cv2.warpPerspective(img1, np.dot(homo_translate, homography), warped_img_shape)

    stitched_img = warped_img.copy()
    stitched_img[-1 * y_min: -1 * y_min + h2, -1 * x_min: -1 * x_min + w2] = img2

    zero_points = np.where(img2 == [0, 0, 0])
    y = zero_points[0] - y_min
    x = zero_points[1] - x_min

    stitched_img[y, x] = warped_img[y, x]

    thresh_img = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(thresh_img, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[len(contours) - 1])
    final_stitched_img = stitched_img[y:y + h, x:x + w]

    return final_stitched_img


def blend_images(img1, img2, FeatureType=1, NumFeatures=1000, MinMatchThresh=10.0, NumIterations=5000, RansacThreshold=5.0, DisplayImgs=True, SaveImgs=False):

    prev_img = img1.copy()
    curr_img = img2.copy()
    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    corners1, corner_mask1 = detect_corners(prev_img, feature_type=FeatureType, num_features=NumFeatures,
                                            display_img=DisplayImgs, save_img=SaveImgs)
    corners2, corner_mask2 = detect_corners(curr_img, feature_type=FeatureType, num_features=NumFeatures,
                                            display_img=DisplayImgs, save_img=SaveImgs)

    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    corners1 = compute_ANMS(prev_img, corners1, corner_mask1, NumFeatures, display_img=DisplayImgs, save_img=SaveImgs)
    corners2 = compute_ANMS(curr_img, corners2, corner_mask2, NumFeatures, display_img=DisplayImgs, save_img=SaveImgs)

    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    feature_dict1 = encode_features(corners1, prev_img, display_img=False, save_img=SaveImgs)
    feature_dict2 = encode_features(corners2, curr_img, display_img=False, save_img=SaveImgs)

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    matched_points = match_features(feature_dict1, feature_dict2, MinMatchThresh)
    show_matching(prev_img, curr_img, matched_points, display_img=DisplayImgs, save_img=SaveImgs)

    """
    Refine: RANSAC, Estimate Homography
    """
    strong_pairs, final_homo = compute_RANSAC(matched_points, NumIterations, RansacThreshold)
    show_matching(prev_img, curr_img, strong_pairs, display_img=DisplayImgs, save_img=SaveImgs, type="RANSAC")

    blended_img = stitch_images(prev_img, curr_img, final_homo)

    if DisplayImgs:
        cv2.namedWindow("blended", cv2.WINDOW_FREERATIO)
        cv2.imshow("blended", blended_img)
        cv2.waitKey(0)

    return blended_img


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=1000, type=int,
                        help='Number of best features to extract from each image, Default:1000')
    Parser.add_argument('--FeatureType', default=1, type=int,
                        help='Type of Feature Detection. 1: shi-tomasi. 0: Harris , Default:1')
    Parser.add_argument('--DataDir', default="..//Data//Train//Set1",
                        help='Location of directory where a set of images are stored, Default: ..//Data//Train//Set1')
    Parser.add_argument('--MatchThresh', default=10.0, help='Threshold for Feature Matching, Default: 10.0')
    Parser.add_argument('--NumIterations', default=5000, help='Iterations to Run RANSAC, Default: 5000')
    Parser.add_argument('--RansacThresh', default=5.0, help='Threshold for Ransac Outlier Rejection, Default: 5.0')
    Parser.add_argument('--DisplayImages', default=True, help='Display the results as they are computed, Default:True')
    Parser.add_argument('--SaveImages', default=False, help='Save the results as they are computed, Default: False')
    Parser.add_argument('--OutDir', default="..//Data//Results", help='Location of directory where results will be stored, Default: ..//Data//Results')

    Args = Parser.parse_args()
    NumFeatures = Args.NumFeatures
    FeatureType = Args.FeatureType
    DataDir = Args.DataDir
    MinMatchThresh = Args.MatchThresh
    NumIterations = Args.NumIterations
    RansacThreshold = Args.RansacThresh
    DisplayImgs = Args.DisplayImages
    SaveImgs = Args.SaveImages
    OutDir = Args.OutDir

    if not os.path.exists(DataDir):
        print("[ERROR]: The following Directory does NOT Exist \n {}".format(DataDir))
    else:
        global curr_out_dir

        curr_out_dir = os.path.join(OutDir, os.path.basename(DataDir))
        if SaveImgs:
            if not os.path.exists(curr_out_dir):
                os.makedirs(curr_out_dir)

        """
        Read a set of images for Panorama stitching
        """
        img_dir = DataDir
        images = [cv2.imread(os.path.join(img_dir, str(i+1) + ".jpg")) for i in range(len(os.listdir(img_dir)))]

        while True:
            i = 0
            num_imgs = len(images)
            blended_images = []
            if num_imgs == 1:
                break
            while i < num_imgs - 1:
                img1 = images[i]
                img2 = images[i + 1]

                blended_img = blend_images(img1, img2, FeatureType, NumFeatures, MinMatchThresh, NumIterations, RansacThreshold, DisplayImgs, SaveImgs)
                blended_images.append(blended_img)
                i += 1
            blended_images.reverse()
            images.clear()
            images = blended_images.copy()

        print('\n')
        print("Final Image")
        cv2.namedWindow("final_image", cv2.WINDOW_FREERATIO)
        cv2.imshow("final_image", images[0])
        cv2.waitKey(0)

        """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """
        if SaveImgs:
            cv2.imwrite(os.path.join(curr_out_dir, "mypano.png"), images[0])


if __name__ == '__main__':
    main()
