import numpy
import cv2
import os
import glob
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
	#grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	grayImage = image

	sift = cv2.SIFT_create()
	#Find interest points.
	keypoints = sift.detect(grayImage, None)
	#Computing features.
	keypoints, features = sift.compute(grayImage, keypoints)
	#Converting keypoints to numbers.
	keypoints = numpy.float32([kp.pt for kp in keypoints])
	return keypoints, features


# 借用 cv2.DescriptorMatcher_create.knnMatch
def cv2_match_keypoints(keypointsL, keypointsR, featuresL, featuresR, thres_ratio=0.7):
    # Slow: featureMatcher = cv2.DescriptorMatcher_create("BruteForce")

    featureMatcher = cv2.DescriptorMatcher_create("FlannBased")
    # use KNN to find top 2 matches.
    # TODO: implement knn
    matches = featureMatcher.knnMatch(featuresL, featuresR, k=2)

    good_matches = []
    for (m,n) in matches:
        # print(m.distance, n.distance)
        if m.distance / n.distance < thres_ratio:
            good_matches.append(m)

    # print(len(good_matches))
    keypointsL = np.uint8([ keypointsL[m.queryIdx] for m in good_matches ]).reshape(-1,2)
    keypointsR = np.uint8([ keypointsR[m.trainIdx] for m in good_matches ]).reshape(-1,2)

    goodMatches_pos = []
    for i in range(len(keypointsL)):
        psL = keypointsL[i]
        psR = keypointsR[i]
        goodMatches_pos.append([psL, psR])
    
    return goodMatches_pos


def match_keypoints(keypointsL, keypointsR, featuresL, featuresR, thres_ratio=0.7):
    Match_idxAndDist = [] # min corresponding index, min distance, seccond min corresponding index, second min distance
    for i in range(len(featuresL)):
        min_IdxDis = [-1, np.inf]  # record the min corresponding index, min distance
        secMin_IdxDis = [-1 ,np.inf]  # record the second corresponding min index, min distance
        for j in range(len(featuresR)):
            dist = np.linalg.norm(featuresL[i] - featuresR[j])
            if (min_IdxDis[1] > dist):
                secMin_IdxDis = np.copy(min_IdxDis)
                min_IdxDis = [j , dist]
            elif (secMin_IdxDis[1] > dist and secMin_IdxDis[1] != min_IdxDis[1]):
                secMin_IdxDis = [j, dist]
        
        Match_idxAndDist.append([min_IdxDis[0], min_IdxDis[1], secMin_IdxDis[0], secMin_IdxDis[1]])

    # ratio test as per Lowe's paper
    goodMatches = []
    for i in range(len(Match_idxAndDist)):
        if (Match_idxAndDist[i][1] <= Match_idxAndDist[i][3] * thres_ratio):
            goodMatches.append((i, Match_idxAndDist[i][0]))
        
    goodMatches_pos = []
    for (idx, correspondingIdx) in goodMatches:
        psA = int(keypointsL[idx][0]), int(keypointsL[idx][1]) # left point
        psB = int(keypointsR[correspondingIdx][0]), int(keypointsR[correspondingIdx][1]) # right point
        goodMatches_pos.append([psA, psB])
    
    # print(goodMatches_pos)
    # print(goodMatches_pos[:,0])
    return goodMatches_pos
    

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


def warping_and_stitching(H, img_L, img_R, image_number):
    # img_R: source
    # img_L: dest
    hl,wl,_ = img_L.shape
    hr,wr,_ = img_R.shape
    stitch_img = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")

    # Warp source (right) image to destinate (left) image by Homography matrix.
    inv_H = np.linalg.inv(H)
    pbar = trange(stitch_img.shape[0])
    pbar.set_description(f"Warping and stitching {image_number}-th images")

    # 從img_L的座標到img_R找對應點 貼到img_L座標
    for i in pbar:
        for j in range(stitch_img.shape[1]):
            coor = np.array([j, i, 1])
            # coor = np.array([i, j, 1])
            # print("H", H)
            projection = np.matmul(inv_H, coor)
            # normalize
            projection = projection / projection[2]
            
            # 
            # TODO: try Nearest neighbors or interpolation to determine color
            y, x = int(round(projection[0])), int(round(projection[1]))    
            # x, y = int(round(projection[0])), int(round(projection[1]))    


            # Boundary test.
            if x < 0 or x >= hr or y < 0 or y >= wr:
                pass
            # else we need the tranform for this pixel.
            else:
                stitch_img[i, j] = img_R[x, y]
    # Blending the result with source (left) image       
    if True:
        cv2.imwrite(f"./report_img/stitch_img_middle_{image_number}.jpg", stitch_img)
        stitch_img = linear_blending(img_L, stitch_img)

    return stitch_img


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

    # # Plot the overlap mask
    if True: 
        # cv2.imwrite(f"./report_img/overlap_mask.jpg", overlap_mask.astype(int), cmap="gray")
        plt.imsave(f"./report_img/overlap_mask.jpg", overlap_mask.astype(int), cmap='gray')

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

    # Put on img_R first
    result_img = np.copy(img_R)
    cv2.imwrite(f"./report_img/blending_img1.jpg", result_img)
    # Put on img_L second
    result_img[:hl, :wl] = np.copy(img_L)
    cv2.imwrite(f"./report_img/blending_img2.jpg", result_img)
    # linear blending img_R
    for i in range(hr):
        for j in range(wr):
            if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                result_img[i, j] = blending_mask[i, j]*img_L[i, j] + (1 - blending_mask[i, j]) * img_R[i, j]
    
    return result_img


def linear_blending_with_const(img_left, img_right):
    (hl, wl) = img_left.shape[:2]
    (hr, wr) = img_right.shape[:2]
    img_left_mask = np.zeros((hr, wr), dtype="int")
    img_right_mask = np.zeros((hr, wr), dtype="int")
    constant_width = 3 # constant width
    
    # find the left image and right image mask region(Those not zero pixels)
    for i in range(hl):
        for j in range(wl):
            if np.count_nonzero(img_left[i, j]) > 0:
                img_left_mask[i, j] = 1
    for i in range(hr):
        for j in range(wr):
            if np.count_nonzero(img_right[i, j]) > 0:
                img_right_mask[i, j] = 1
                
    # find the overlap mask(overlap region of two image)
    overlap_mask = np.zeros((hr, wr), dtype="int")
    for i in range(hr):
        for j in range(wr):
            if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                overlap_mask[i, j] = 1
    
    # compute the alpha mask to linear blending the overlap region
    alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
    for i in range(hr):
        minIdx = maxIdx = -1
        for j in range(wr):
            if (overlap_mask[i, j] == 1 and minIdx == -1):
                minIdx = j
            if (overlap_mask[i, j] == 1):
                maxIdx = j

            
        if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
            continue
            
        decrease_step = 1 / (maxIdx - minIdx)
        
        # Find the middle line of overlapping regions, and only do linear blending to those regions very close to the middle line.
        middleIdx = int((maxIdx + minIdx) / 2)
        
        # left 
        for j in range(minIdx, middleIdx + 1):
            if (j >= middleIdx - constant_width):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
            else:
                alpha_mask[i, j] = 1
        # right
        for j in range(middleIdx + 1, maxIdx + 1):
            if (j <= middleIdx + constant_width):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
            else:
                alpha_mask[i, j] = 0

    linearBlendingWithConstantWidth_img = np.copy(img_right)
    linearBlendingWithConstantWidth_img[:hl, :wl] = np.copy(img_left)
    # linear blending with constant width
    for i in range(hr):
        for j in range(wr):
            if (np.count_nonzero(overlap_mask[i, j]) > 0):
                linearBlendingWithConstantWidth_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
    
    return linearBlendingWithConstantWidth_img


def remove_black_border(img):
    h,w,_=img.shape
    reduced_h, reduced_w = h, w
    # right to left
    for col in range(w-1, -1, -1):
        all_black = True
        for i in range(h):
            if np.count_nonzero(img[i, col]) > 0:
                all_black = False
                break
        if (all_black == True):
            reduced_w = reduced_w - 1
            
    # bottom to top 
    for row in range(h - 1, -1, -1):
        all_black = True
        for i in range(reduced_w):
            if np.count_nonzero(img[row, i]) > 0:
                all_black = False
                break
        if (all_black == True):
            reduced_h = reduced_h - 1
    
    return img[:reduced_h, :reduced_w] 


def load_data_and_f(data_name):
    root = os.getcwd()
    img_path = os.path.join(root, 'data_small', data_name,"*.JPG")
    filenames = sorted(glob.glob(img_path))
    images = []
    for fn in tqdm(filenames):
        images.append(cv2.imread(fn))

    focal_path = os.path.join(root, 'data_small', data_name,"focal.txt")
    with open(focal_path, "r") as f:
        focals = f.read().splitlines()
    focals = sorted(focals)
    focals = [float(f.split(" ")[1]) for f in focals] #

    return images, focals, filenames


def cyclindrical(img, f):
    h, w = img.shape[0], img.shape[1]
    proj = np.zeros(img.shape, np.uint8)
    print('Run projection')
    for x in tqdm(range(-w//2, w-w//2)):
        for y in range(-h//2, h-h//2):
            proj_x = round(f*math.atan(x/f)) + w//2
            proj_y = round(f*y/math.sqrt(x**2 + f**2)) + h//2
            proj[proj_y][proj_x] = img[y + h//2][x + w//2]
    return proj

def load_data(data_name):
    root = os.getcwd()
    img_path = os.path.join(root, 'data_small', data_name,"*.JPG")
    #img_path = os.path.join(root, 'data_small', data_name,"*.jpg")
    filenames = sorted(glob.glob(img_path))
    images = []
    for fn in tqdm(filenames):
        images.append(cv2.imread(fn))

    return images, filenames


def draw_matches(img_left, img_right, left_pos, right_pos):
        '''
            Draw the match points img with keypoints and connection line
        '''
        
        # initialize the output visualization image
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        vis = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")
        vis[0:hl, 0:wl] = img_left
        vis[0:hr, wl:] = img_right
        
        # Draw the match

        for i in range(len(left_pos)):
            pos_l = left_pos[i]
            pos_r = right_pos[i][0] + wl, right_pos[i][1]
            cv2.circle(vis, pos_l, 3, (0, 0, 255), 1)
            cv2.circle(vis, pos_r, 3, (0, 255, 0), 1)
            cv2.line(vis, pos_l, pos_r, (255, 0, 0), 1)

        # return the visualization
        cv2.imwrite("./report_img/matching.jpg", vis)
        
        return vis


if __name__=="__main__":

    fixed_random(2322)
    root = os.getcwd()
    images, image_paths = load_data("data1")
    # images, focals, image_paths = load_data_and_f("data1")
    # print(image_paths)
    os.makedirs("report_img", exist_ok=True) 
    final_img = images[0]
    # final_img = cyclindrical(images[0], focals[0]) 
    img_name = image_paths[0].split("/")[-1]
    cv2.imwrite(f"./report_img/{img_name}", final_img)

    for i in trange(1, len(images)):
        # stitch image R(src) to image L(dst)
        img_L = final_img
        img_R = images[i]
        # img_R = cyclindrical(images[i], focals[i]) 
        

        img_name = image_paths[i].split("/")[-1]
        cv2.imwrite(f"./report_img/{img_name}", img_R)
        print("Find feature")
        print("Describe feature")
        keypointsL, featuresL = find_and_describe_features(img_L)
        keypointsR, featuresR = find_and_describe_features(img_R)

        print("match feature")
        # matches: (featuresL, featuresR)

        # matches_pos = cv2_match_keypoints(keypointsL, keypointsR, featuresL, featuresR, thres_ratio=0.7)
        matches_pos = match_keypoints(keypointsL, keypointsR, featuresL, featuresR, thres_ratio=0.7)
        
        src_pts = []
        dst_pts = []
        for PointL, PointR in matches_pos: 
            src_pts.append(list(PointR)) # src: imgR, dst: imgL
            dst_pts.append(list(PointL)) 
        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)
        
        if True: # and Draw
            draw_matches(img_L, img_R, dst_pts, src_pts)
            #draw_matches(img_L, img_R, matches_pos)


        print("image matching: map left to right")
        # best_H = RANSAC_find_homography(src_pts, dst_pts)
        best_H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=8000)
        # print(best_H)
        # print(M)

        # print("image warping and stitching")
        final_img = warping_and_stitching(best_H, final_img, img_R, i)

        # print("image warping")
        # # compute offest based on given H.
        # warped_image_src, src_offset_x, src_offset_y = warping(img_R, best_H)
        # cv2.imwrite(f"./report_img/warp_image_{i}.jpg", warped_image_src)

        # print("image stitching")
        # final_img = stitching(final_img, warped_image_src, src_offset_x, src_offset_y, matches)

        print("Cut the black row")
        final_img = remove_black_border(final_img)
        print("Saving result.")
        print(final_img.shape)
        cv2.imwrite(f"./report_img/stitch_image_{i}.jpg", final_img)


    print("Done.")
    cv2.imwrite(f"./report_img/result.jpg", final_img)
