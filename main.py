import numpy
import cv2
import os
from glob import glob
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import random
import math

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
    # TODO: implement knn
    matches = featureMatcher.knnMatch(featuresL, featuresR, k=2)

    # print(len(matches))

    # store all the good matches as per Lowe's ratio test.
    # filter out some poor matches: RANSAC to select a set of inliers 
    # 找出最好以及第二好的點，比較他們間的距離比值(ratio)
    # ||f1 — f2 || / || f1 — f2’ ||，如果他們很像(ratio 就會趨近於1)則不使用此點
    good_matches = []
    for (m,n) in matches:
        # print(m.distance, n.distance)
        if m.distance / n.distance < thres_ratio:
            good_matches.append(m)

    # for m in good_matches:
    #     print(m.queryIdx, m.trainIdx, m.distance)

    # print(len(good_matches))
    return good_matches #, good_matches_pos


# https://yungyung7654321.medium.com/python%E5%AF%A6%E4%BD%9C%E8%87%AA%E5%8B%95%E5%85%A8%E6%99%AF%E5%9C%96%E6%8B%BC%E6%8E%A5-automatic-panoramic-image-stitching-28629c912b5a
def solve_H(P, M):
    # P: points in the original (img_R) plane,
    # M: points in the target (img_L) plane. 
    A = []  
    for i in range(len(P)): 
        x1, y1 = P[i][0], P[i][1]
        x2, y2 = M[i][0], M[i][1]
        A.append([x1, y1, 1, 
                  0, 0, 0, 
                  -x1*x2, -y1*x2, -x2])
        A.append([0, 0, 0, 
                  x1, y1, 1, 
                  -x1*y2, -y1*y2, -y2])
    # Solve linear equations Ah = 0 using SVD
    u, s, v = np.linalg.svd(A) 
    # pick H from last line of vt  
    H = np.reshape(v[8], (3,3))
    # normalization: let H[2,2] equals to 1
    H = H / (H.item(8))

    return H

def RANSAC_find_homography(src_pts, dst_pts, num_iter=8000, threshold=5, 
                           num_sample=4):
    # src_pts: from img_R
    # dst_pts: from img_L 

    # Solve homography matrix 
    # P: original plane (img_R)
    # M: target plane (img_L)
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
    # img_R: source
    # img_L: dest
    hl,wl,_ = img_L.shape
    hr,wr,_ = img_R.shape
    stitch_img = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")


    # Warp source (right) image to destinate (left) image by Homography matrix.
    pbar = trange(stitch_img.shape[0])
    pbar.set_description(f"Warping and stitching {i}-th images")
    inv_H = np.linalg.inv(H)
    for i in pbar:
        for j in range(stitch_img.shape[1]):
            coor_L = np.array([j, i, 1])
            # print("H", H)
            projection_L2R = np.matmul(inv_H, coor_L)
            # normalize
            projection_L2R = projection_L2R / projection_L2R[2]
            
            # Nearest neighbors 
            # TODO: try interpolation
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

    return stitch_img

# def warping(img, H):
#     h,w,_ = img.shape
#     top_left = np.array([0, 0, 1])
#     top_right = np.array([w, 0, 1])
#     bot_left = np.array([0, h, 1])
#     bot_right = np.array([w-1, h-1, 1])

#     new_top_left = np.dot(H, top_left)
#     new_top_right = np.dot(H, top_right)
#     new_bot_left = np.dot(H, bot_left)
#     new_bot_right = np.dot(H, bot_right)

#     new_top_left = new_top_left/new_top_left[2]
#     new_top_right = new_top_right/new_top_right[2]
#     new_bot_left = new_bot_left/new_bot_left[2]
#     new_bot_right = new_bot_right/new_bot_right[2]

#     # print(new_top_left, new_top_right, new_bot_left, new_bot_right)
#     x_values = [new_top_left[0], new_top_right[0], new_bot_left[0], new_bot_right[0]]
#     y_values = [new_top_left[1], new_top_right[1], new_bot_left[1], new_bot_right[1]]
#     offset_x = math.floor(min(x_values))
#     offset_y = math.floor(min(y_values))
#     print(offset_x, offset_y)
#     max_x = math.ceil(max(x_values))
#     max_y = math.ceil(max(y_values))  
    
#     size_x = max_x - offset_x
#     size_y = max_y - offset_y
#     print("size_x: ", size_x, "\t size_y: ", size_y)

#     H_inv = np.linalg.inv(H)
#     warped_image = np.zeros((size_y, size_x, 3))

#     for x in range(size_x):
#         for y in range(size_y):
#             new_x, new_y, z = np.dot(H_inv, [[x+offset_x], [y+offset_y], [1]])
#             # normalize
#             new_x = int(new_x/z)
#             new_y = int(new_y/z)
#             # if in boundary
#             if new_x >= 0 and new_x < w and new_y >= 0 and new_y < h:
#                 warped_image[y, x, :] = img[new_y, new_x, :]

#     return warped_image, offset_x, offset_y 


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
        cv2.imwrite(f"./report_img/overlap_mask.jpg", overlap_mask.astype(int))

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

# def padding_img(img, move_x, move_y):
#     if(move_x >= 0 and move_y >= 0):
#         new_img = np.pad(img, ((move_y, 0), (move_x, 0), (0, 0)), 'constant')
#     elif(move_x >= 0 and move_y < 0):
#         new_img = np.pad(img, ((0, -move_y), (move_x, 0), (0, 0)), 'constant')
#     elif(move_x < 0 and move_y >= 0):
#         new_img = np.pad(img, ((move_y, 0), (0, -move_x), (0, 0)), 'constant')
#     else:
#         new_img = np.pad(img, ((0, -move_y), (0, -move_x), (0, 0)), 'constant')
#     return new_img

# def stitching(img_L, img_R, offset_x_R, offset_y_R, matches):

#     if offset_x_R<0: 
#         print("inv")
#         offset_x_R = -offset_x_R
#         offset_y_R = -offset_y_R
#         matches = matches[1], matches[0]
#         img_L, img_R = img_R, img_L

#     hl,wl,_ = img_L.shape
#     hr,wr,_ = img_R.shape

#     img_L_padding_x = wr - wl + (matches[0][0] - matches[1][0])
#     img_R_padding_x = (matches[0][0] - matches[1][0])
#     intersect_range = wl[1] + (matches[1][0] - matches[0][0]) 
    
#     new_img1 = padding_img(img_L, -img_L_padding_x, -offset_y_R)
#     new_img2 = padding_img(img_R, img_R_padding_x, offset_y_R)
    
#     print(new_img1.shape)
#     result = (np.zeros(new_img1.shape))
    
#     # linear blending
#     intersect_cnt = 0
#     for i in range(0, result.shape[1]):

#         origin_percent = 1
#         new_percent = 1

#         if (np.count_nonzero(new_img1[:,i][:] != 0)) and (np.count_nonzero(new_img2[:,i][:] != 0)):
#             origin_percent = (1 - intersect_cnt / intersect_range)
#             new_percent = (intersect_cnt / intersect_range)
#             intersect_cnt += 1
#         elif(np.count_nonzero(new_img1[:,i][:] != 0)):
#             origin_percent = 1
#             new_percent = 0
#         elif(np.count_nonzero(new_img2[:,i][:] != 0)):
#             origin_percent = 0
#             new_percent = 1

#         result[:,i][:] = origin_percent * new_img1[:,i][:]  + new_percent* new_img2[:,i][:]  
        
#     return result



if __name__=="__main__":

    fixed_random(2322)
    root = os.getcwd()
    # image_paths = glob(os.path.join(root,"data","data1","*.JPG"))
    image_paths = sorted(glob(os.path.join(root,"data_small","data1","*.JPG")))
    # print(os.path.join(root,"data1","*.JPG"))
    # print(image_paths)
    os.makedirs("report_img", exist_ok=True)

    final_img = cv2.imread(image_paths[0])
    img_name = image_paths[0].split("/")[-1]
    cv2.imwrite(f"./report_img/{img_name}", final_img)

    for i in trange(1, len(image_paths)):
        # stitch image R(src) to image L(dst)
        img_L = final_img
        img_R = cv2.imread(image_paths[i])

        img_name = image_paths[i].split("/")[-1]
        cv2.imwrite(f"./report_img/{img_name}", img_R)
        print("Find feature")
        print("Describe feature")
        keypointsL, featuresL = find_and_describe_features(img_L)
        keypointsR, featuresR = find_and_describe_features(img_R)

        print("match feature")
        # matches: (featuresL, featuresR)
        matches = match_features(featuresL, featuresR, thres_ratio=0.7)
        print("The number of matching points:", len(matches))
        

        dst_pts = np.uint8([ keypointsL[m.queryIdx] for m in matches ]).reshape(-1,2)
        src_pts = np.uint8([ keypointsR[m.trainIdx] for m in matches ]).reshape(-1,2)

        # if i==0: # and Draw
        #     draw_matches(img_L, img_R, src_pts, dst_pts, number=i)

        print("image matching: map left to right")
        # best_H = RANSAC_find_homography(src_pts, dst_pts)
        best_H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=8000)
        # print(best_H)
        # print(M)

        print("image warping and stitching")
        final_img = warping_and_stitching(best_H, final_img, img_R, i)

        # print("image warping")
        # # compute offest based on given H.
        # warped_image_src, src_offset_x, src_offset_y = warping(img_R, best_H)
        # cv2.imwrite(f"./report_img/warp_image_{i}.jpg", warped_image_src)

        # print("image stitching")
        # final_img = stitching(final_img, warped_image_src, src_offset_x, src_offset_y, matches)

        print("Saving result.")
        print(final_img.shape)
        cv2.imwrite(f"./report_img/stitch_image_{i}.jpg", final_img)
        break

    print("Done.")
    # cv2.imwrite(f"./report_img/warp_image.jpg", final_img)
