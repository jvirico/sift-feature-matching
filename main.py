import cv2 as cv
import numpy as np

## DONE:
#   (1) Load two frames of a scene
#       - We convert to gray scale to ease computation and visualize
#   (2) Compute PoI Detection and Description
#       - We compute PoI Detection using SIFT and PoI Description also using SIFT, 
#         alternatively we could choose a different method for PoI detection and PoI description
#         such as SIFT, SURF, KAZE, Difference of Hessian, K.. (Detectors), 
#         and SIFT, SURF, DSP-SIFT, KAZE... (Descriptors)
#   (3) Look for PoI matching in both frames [1]
#       - We can use different methods, e.g. (3.1) a brute force approach using cv2.BFMatcher,
#         or (3.2) a nearest neighbour approach FLANN, using cv2.FlannBasedMatcher.

# image resize factor, to ease computations (0.1 = 10% of original size), 
resize_factor = 0.1 # use 1 to use full image spatial resolution
show_n_matches = 20

def FilterMatches(matches, mode='selection'):
    '''
        Implements different options to filter matches.
            - 'sort' will simply sort them by similarity distance.
            - 'ratio' will employ the ratio test as in Lowe's paper
    '''
    if(mode == 'sort'):
        selection = sorted(matches, key = lambda x:x[0].distance)
    if(mode == 'ratio'):
        selection = [[m] for m, n in matches if m.distance < 0.7*n.distance]

    return selection

def Run_SIFT(img1,img2):
    '''
        Helper function to compute feature detection and descrition using SIFT-SIFT
    '''
    # Create a SIFT detector
    sift = cv.SIFT_create()

    ## Key point Detection and Description using SIFT-SIFT
    # The second parameter is a mask, since we want to use the whole image to look for matches, we pass no mask.
    k1, d1 = sift.detectAndCompute(img1, None) # returns: keypoints (PoIs), descriptors
    k2, d2 = sift.detectAndCompute(img2, None)

    return k1,k2,d1,d2

## (1)
img1 = cv.imread('data/Scene2/20200613_181153.jpg')
#img2 = cv.imread('data/Scene2/20200613_181156.jpg')
img2 = cv.imread('data/Scene2/20200613_181645.jpg')

print(img1.shape)

img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# we resize to ease visualization
'''
img1_g_small = cv.resize(img1_gray,(int(img1_gray.shape[1]*0.2),int(img1_gray.shape[0]*0.2)))
cv.imshow('frame1',img1_g_small)
cv.waitKey(0)
'''
## (2)
img1_gs = cv.resize(img1_gray,(int(img1_gray.shape[1]*resize_factor),int(img1_gray.shape[0]*resize_factor)))
img2_gs = cv.resize(img2_gray,(int(img2_gray.shape[1]*resize_factor),int(img2_gray.shape[0]*resize_factor)))

keypoints1,keypoints2,descriptors1,descriptors2 = Run_SIFT(img1_gs,img2_gs)


## (3) Feature Matching
# (3.1) Using cv2.BFMatcher
bf = cv.BFMatcher()
# We match descriptors using Brute-Force Matching and sort them by similarity distance
matches = bf.knnMatch(descriptors1,descriptors2,k=2)
print('BFMatcher')
print('Found %s unfiltered matches'% len(matches))
matches = FilterMatches(matches,mode='ratio')
print('Selected %s matches.'% len(matches))
# We draw first 9 matches
img_matches = img1
img_matches = cv.drawMatchesKnn(img1=img1_gs,img2=img2_gs,keypoints1=keypoints1,keypoints2=keypoints2,
                                matches1to2=matches[:show_n_matches],outImg=None,
                                matchColor=(0, 255, 0), matchesMask=None,
                                singlePointColor=(255, 0, 0), flags=0)
cv.imshow('BFMatcher results',img_matches)
cv.waitKey(0)

# (3.2) Using cv.FlannBasedMatcher
# FLANN index and search parameters to be used appropiate to SIFT
FLANN_INDEX_KDTREE = 1
idx_prms = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) # index parameters
srch_prms = dict(checks=50) # search parameters, it could be an empty dictionary
# srch_prms = {}
flann = cv.FlannBasedMatcher(idx_prms,srch_prms)
matches = flann.knnMatch(descriptors1,descriptors2,k=2)
# mask to filter good matches
#good_matches = [[0,0] for i in range(len(matches))]
print('FLANN Matcher')
print('Found %s unfiltered matches'% len(matches))
matches = FilterMatches(matches,mode='ratio')
print('Selected %s matches.'% len(matches))
# We draw first show_n_matches matches
img_matches = img1
img_matches = cv.drawMatchesKnn(img1=img1_gs,img2=img2_gs,keypoints1=keypoints1,keypoints2=keypoints2,
                                matches1to2=matches[:show_n_matches],outImg=None,
                                matchColor=(0, 255, 0), matchesMask=None,
                                singlePointColor=(255, 0, 0), flags=0)
cv.imshow('FLANN results',img_matches)
cv.waitKey(0)






