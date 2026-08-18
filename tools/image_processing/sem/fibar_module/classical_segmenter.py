# CODE FOR the Classical segmentation approach # 

# libraries
import numpy as np
import cv2 
import math
from pykuwahara import kuwahara


# Huang function directly adopted from https://github.com/dnhkng/Huang-Thresholding/blob/master/analysis.py #

def huang(data):
    """Implements Huang's fuzzy thresholding method 
        Uses Shannon's entropy function (one can also use Yager's entropy function) 
        Huang L.-K. and Wang M.-J.J. (1995) "Image Thresholding by Minimizing  
        the Measures of Fuzziness" Pattern Recognition, 28(1): 41-51"""
    threshold = -1
    first_bin = 0
    for ih in range(254):
        if data[ih] != 0:
            first_bin = ih
            break
    last_bin = 254
    for ih in range(254, -1, -1):
        if data[ih] != 0:
            last_bin = ih
            break
    term = 1.0 / (last_bin - first_bin)
    mu_0 = np.zeros(shape=(254, 1))
    num_pix = 0.0
    sum_pix = 0.0
    for ih in range(first_bin, 254):
        sum_pix += (ih * data[ih])
        num_pix += data[ih]
        mu_0[ih] = sum_pix / num_pix 
    mu_1 = np.zeros(shape=(254, 1))
    num_pix = 0.0
    sum_pix = 0.0
    for ih in range(last_bin, 1, -1):
        sum_pix += (ih * data[ih])
        num_pix += data[ih]
        mu_1[ih - 1] = sum_pix / num_pix  
    min_ent = float("inf")
    for it in range(254):
        ent = 0.0
        for ih in range(it):
            # Equation (4) in Reference
            mu_x = 1.0 / (1.0 + term * math.fabs(ih - mu_0[it]))
            if not ((mu_x < 1e-06) or (mu_x > 0.999999)):
                # Equation (6) & (8) in Reference
                ent += data[ih] * (-mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x))
        for ih in range(it + 1, 254):
            # Equation (4) in Ref. 1 */
            mu_x = 1.0 / (1.0 + term * math.fabs(ih - mu_1[it]))
            if not ((mu_x < 1e-06) or (mu_x > 0.999999)):
                # Equation (6) & (8) in Reference
                ent += data[ih] * (-mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x))
        if ent < min_ent:
            min_ent = ent
            threshold = it
 
    return threshold

def classical_segment(file):
    """
    Main function for segmenting the original input image classically 
    """
    img_data = cv2.imread(file, 0)

    # kuwahara filter 
    kuwahara_img = kuwahara(img_data, method='mean', radius=2)

    histogram, _ = np.histogram(kuwahara_img, bins=range(256))
    huang_th = huang(histogram)
    np_thuang = np.where(img_data > huang_th, 1, 0)
    tmp_h = np_thuang
    tmp_h[tmp_h == 1] = 255

    merged = np.uint8(cv2.medianBlur(np.uint8(np_thuang), 15)) 
    merged[merged>0] = 255

    # return segmented im 
    return np.uint8(merged)
