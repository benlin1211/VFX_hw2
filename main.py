import numpy
import cv2
import os
from glob import glob
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import random

def fixed_random(seed=2322):
    random.seed(seed)
    np.random.seed(seed)

# 借用 cv2
def find_and_describe_features(image):
	#Getting gray image
	grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#Find and describe the features.
	# Fast: sift = cv2.xfeatures2d.SURF_create()
	# sift = cv2.xfeatures2d.SIFT_create()
	sift = cv2.SIFT_create()
	#Find interest points.
	keypoints = sift.detect(grayImage, None)
	#Computing features.
	keypoints, features = sift.compute(grayImage, keypoints)
	#Converting keypoints to numbers.
	keypoints = numpy.float32([kp.pt for kp in keypoints])
	return keypoints, features


# 借用 cv2.DescriptorMatcher_create.knnMatch
def match_features(featuresL, featuresR, thres_ratio=0.7):
    # Slow: featureMatcher = cv2.DescriptorMatcher_create("BruteForce")

    featureMatcher = cv2.DescriptorMatcher_create("FlannBased")
    # use KNN to find top 2 matches.
    matches = featureMatcher.knnMatch(featuresL, featuresR, k=2)

    # ransac
    # store all the good matches as per Lowe's ratio test.
    # filter out some poor matches: RANSAC to select a set of inliers 
    # 找出最好以及第二好的點，比較他們間的距離比值(ratio)
    # ||f1 — f2 || / || f1 — f2’ ||，如果他們很像(ratio 就會趨近於1)則不使用此點
    good_matches = []
    for m,n in matches:
        if m.distance < n.distance * thres_ratio:
        # if True:
            good_matches.append(m)
    return good_matches #, good_matches_pos


# def draw_matches(img_L, img_R, src_pts, dst_pts, number):

#     hl,wl,_ = img_L.shape
#     hr,wr,_ = img_R.shape

#     img_match = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")
#     img_match[0:hl, 0:wl] = img_L
#     img_match[0:hr, wl:] = img_R
#     # Draw the match
#     for i, _ in enumerate(src_pts):
#         pos_l = src_pts[i]
#         pos_r = dst_pts[i][0]+wl, dst_pts[i][1]
#         cv2.circle(img_match, pos_l, 3, (0, 0, 255), 2) # src 
#         cv2.circle(img_match, pos_r, 3, (0, 255, 0), 2) # dest
#         cv2.line(img_match, pos_l, pos_r, (255, 0, 0), 2)
    
#     os.makedirs("report_img", exist_ok=True)
#     cv2.imwrite(f".report_img/matching_{number}.jpg", img_match)

# https://yungyung7654321.medium.com/python%E5%AF%A6%E4%BD%9C%E8%87%AA%E5%8B%95%E5%85%A8%E6%99%AF%E5%9C%96%E6%8B%BC%E6%8E%A5-automatic-panoramic-image-stitching-28629c912b5a
def solve_H(P, M):
    A = []  
    for r in range(len(P)): 
        A.append([-P[r,0], -P[r,1], -1, 
                  0, 0, 0, 
                  P[r,0]*M[r,0], P[r,1]*M[r,0], M[r,0]])
        A.append([0, 0, 0, 
                  -P[r,0], -P[r,1], -1, 
                  P[r,0]*M[r,1], P[r,1]*M[r,1], M[r,1]])
    # Solve linear equations Ah = 0 using SVD
    u, s, vt = np.linalg.svd(A) 
    # pick H from last line of vt  
    H = np.reshape(vt[8], (3,3))
    # normalization: let H[2,2] equals to 1
    H = (1/H.item(8)) * H

    return H

def RANSAC_find_homography(src_pts, dst_pts, num_iter=8000, threshold=15, 
                           num_sample=4):
    # src_pts: from img_R
    # dst_pts: from img_L 

    # Solve homography matrix 
    # P: original plane
    # M: target plane
    assert num_sample>=4, "We need 4 freedom degrees."
    num_pts = src_pts.shape[0] 
    max_inlier=0
    best_H = None

    # Riun k times to find best Homography with maximum number of inlier.
    for k in range(num_iter):
        # Draw n samples randomly.
        sample_idx = random.sample(range(num_pts), num_sample) 
        #  Compute Homography matrix based on sample points
        H = solve_H(src_pts[sample_idx], dst_pts[sample_idx])
        num_inlier = 0 
        # Compute score of inlier within a threshold in this round
        for i in range(num_pts):
            # Compute distance of inliers & outliers
            if i not in sample_idx:
                src_coor = np.hstack((src_pts[i], [1])) # (x, y, 1)
                # calculate the coordination after transform to destination img 
                projection = np.matmul(H, src_coor.T) 

                #print(i, ":",projection)
                if projection[2] <= 1e-7: 
                    # Small z coornidate value will result in large number after normalization.
                    # => It is not going to be inlier.
                    pass
                else:
                    # Normalize
                    projection = projection / projection[2]
                    #print(projection[:2], dst_pts[i])
                    #print("np.linalg.norm(projection[:2] - dst_pts[i]", np.linalg.norm(projection[:2] - dst_pts[i])) 
                    if (np.linalg.norm(projection[:2] - dst_pts[i]) < threshold):
                        num_inlier = num_inlier + 1
        # Get maximum voting score
        if (max_inlier < num_inlier):
            max_inlier = num_inlier
            best_H = H
            print(f"iter {k}: max_inlier", max_inlier)
    print("The Number of Maximum Inlier:", max_inlier)
    return best_H


def warping_and_stitching(H, img_L, img_R, i):
    
    hl,wl,_ = img_L.shape
    hr,wr,_ = img_R.shape
    stitch_img = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")


    # Warp source (right) image to destinate (left) image by Homography matrix.
    pbar = trange(stitch_img.shape[0])
    pbar.set_description(f"Warping and stitching {i}-th images")
    for i in pbar:
        for j in range(stitch_img.shape[1]):
            coor_L = np.array([i, j, 1])
            # print("H", H)
            projection_L2R = np.matmul(np.linalg.inv(H), coor_L)
            # normalize
            projection_L2R = projection_L2R / projection_L2R[2]
            
            # Nearest neighbors 
            # TODO: try  interpolation
            y, x = int(round(projection_L2R[0])), int(round(projection_L2R[1]))    
            
            # Boundary test.
            if x < 0 or x >= hr or y < 0 or y >= wr:
                pass
            # else we need the tranform for this pixel.
            else:
                stitch_img[i, j] = img_R[x, y]
    # Blending the result with source (left) image       
    if True:
        stitch_img = linear_blending(img_L, stitch_img)

    return final_img


def linear_blending(img_L, img_R):
    hl,wl,_ = img_L.shape
    hr,wr,_ = img_R.shape

    # find non-zero pixel in left image and right image 
    img_L_mask = np.zeros((hr, wr), dtype="int") # non-zero: 1
    img_R_mask = np.zeros((hr, wr), dtype="int") # non-zero: 1
    for i in range(hl):
        for j in range(wl):
            if np.count_nonzero(img_L[i, j]) > 0:
                img_L_mask[i, j] = 1
    for i in range(hr):
        for j in range(wr):
            if np.count_nonzero(img_R[i, j]) > 0: 
                img_R_mask[i, j] = 1 
                
    # find the overlap mask between image L and image R.
    overlap_mask = np.zeros((hr, wr), dtype="int")
    for i in range(hr):
        for j in range(wr):
            if (np.count_nonzero(img_L_mask[i, j]) > 0 and np.count_nonzero(img_R_mask[i, j]) > 0):
                overlap_mask[i, j] = 1
    # Plot the overlap mask
    if True: 
        os.makedirs("report_img", exist_ok=True)
        cv2.imwrite(f".report_img/overlap_mask.jpg", overlap_mask.astype(int))

    blending_mask = np.zeros((hr, wr))    
    for i in range(hr): 
        # find max idx
        minIdx = -1
        maxIdx = -1
        for j in range(wr):
            if (overlap_mask[i, j] == 1 and minIdx == -1):
                minIdx = j
            if (overlap_mask[i, j] == 1):
                maxIdx = j
        if (minIdx == maxIdx): 
            # represent this row's pixels are all zero, or only one pixel not zero
            pass
        else:
            step = 1 / (maxIdx - minIdx)
            # linear interpolation
            for j in range(minIdx, maxIdx + 1):
                blending_mask[i, j] = 1 - (step * (j - minIdx))

    result_img = np.copy(img_R)
    result_img[:hl, :wl] = np.copy(img_L)
    # linear blending
    for i in range(hr):
        for j in range(wr):
            if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                result_img[i, j] = blending_mask[i, j]*img_L[i, j] + (1 - blending_mask[i, j]) * img_R[i, j]
    
    return result_img


if __name__=="__main__":

    fixed_random(2322)
    root = os.getcwd()
    # image_paths = glob(os.path.join(root,"data","data1","*.JPG"))
    image_paths = glob(os.path.join(root,"data_small","data1","*.JPG"))
    # print(os.path.join(root,"data1","*.JPG"))
    # print(image_paths)
    
    final_img = cv2.imread(image_paths[0])
    for i in trange(1, len(image_paths)):
        # stitch image R(src) to image L(dst)
        img_L = final_img
        img_R = cv2.imread(image_paths[i])

        print("Find feature")
        print("Describe feature")
        keypointsL, featuresL = find_and_describe_features(img_L)
        keypointsR, featuresR = find_and_describe_features(img_R)

        print("match feature")
        # matches = match_features(featuresL, featuresR)
        matches = match_features(featuresL, featuresR, thres_ratio=0.7)
        print("The number of matching points:", len(matches))
        
        dst_pts = np.uint8([ keypointsL[m.queryIdx] for m in matches ]).reshape(-1,2)
        src_pts = np.uint8([ keypointsR[m.trainIdx] for m in matches ]).reshape(-1,2)

        # if i==0: # and Draw
        #     draw_matches(img_L, img_R, src_pts, dst_pts, number=i)

        print("image matching: map left to right")
        best_H = RANSAC_find_homography(src_pts, dst_pts)

        print("image warping and stitching")
        final_img = warping_and_stitching(best_H, final_img, img_R, i)

        print("Saving result.")
        os.makedirs("report_img", exist_ok=True)
        cv2.imwrite(f".report_img/warp_image{i}.jpg", final_img)

    print("Done.")
    os.makedirs("report_img", exist_ok=True)
    cv2.imwrite(f".report_img/warp_image.jpg", final_img)
