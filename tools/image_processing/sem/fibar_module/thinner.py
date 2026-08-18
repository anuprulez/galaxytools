# CODE for obtaining the distance transform and thinned/skeletonized image #

# libraries
from plantcv import plantcv as pcv
import cv2 
import numpy as np
import skimage.morphology 

def thinner(PATH_1):
     """
     Function for obtaining the distance transform and skeletonized image from the segmented image (10k, 15k, 20k magnification)
     """
     dist = cv2.distanceTransform(PATH_1, cv2.DIST_L2, 3)
     
     thinned = skimage.morphology.medial_axis(PATH_1).astype(np.uint8)
     thinned[thinned == 1] = 255

     #### pre-processing of the thinned image ####

     # removing skeleton hairs/redundant branches - https://plantcv.readthedocs.io/en/stable/prune/
     pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=thinned, size=50)
     thinned = np.uint8(pruned_skeleton)

     return dist, thinned

def thinner_2k_5k(PATH_1):
     """
     Function for obtaining the distance transform and skeletonized image from the segmented image (2k, 5k magnification)
     """
     dist = cv2.distanceTransform(PATH_1, cv2.DIST_L2, 3)
     
     thinned = skimage.morphology.medial_axis(PATH_1).astype(np.uint8)
     thinned[thinned == 1] = 255

     return dist, thinned
