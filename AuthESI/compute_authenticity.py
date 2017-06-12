"""
Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity
Ning Zhang, Lin Zhang*, and Zaixi Cheng
"""

import cv2
import numpy as np
import os
import math
from PIL import Image
import guided_filter
import compute_aggd
import scipy.io as sio
import warnings

# warnings.simplefilter("error")
# warnings.simplefilter("ignore", category=RuntimeWarning)


BLOCK_SIZE_ROW     = 48
BLOCK_SIZE_COL     = 48
NORMALIZED_WIDTH   = 528
FEATURE_NUMBER     = 16
GRADIENT_THRESHOLD_L = 3
GRADIENT_THRESHOLD_R = 60
DARK_CHANNEL_THRESHOLD_L = 30
DARK_CHANNEL_THRESHOLD_R = 100


def authenticity(img):
    data = sio.loadmat('prisparam_16_hazeandfog.mat')
    mu_prisparam1 = data['mu1']
    mu_prisparam2 = data['mu2']
    cov_prisparam1 = data['cov1']
    cov_prisparam2 = data['cov2']


    img = cv2.resize(cv2.imread(img), (NORMALIZED_WIDTH, NORMALIZED_WIDTH), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    block_rownum = math.floor(gray.shape[0]/BLOCK_SIZE_ROW)
    block_colnum = math.floor(gray.shape[1]/BLOCK_SIZE_COL)
    img = img[:block_rownum*BLOCK_SIZE_ROW, :block_colnum*BLOCK_SIZE_COL, :]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]

    # gradient magnitude
    gradx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grady = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    absX = cv2.convertScaleAbs(gradx)
    absY = cv2.convertScaleAbs(grady)
    gradient = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    gradient2 = gradient
    gradient2[gradient2 < 20] = 0
    gradient2[gradient2 >= 20] =255

    # dark channel
    dark_image = np.asarray(guided_filter.getDark(Image.fromarray(np.uint8(img)), guided_filter.minimizeFilter, (10, 10)))

    quality = []
    features1_list_all = []  # haze-free features
    features2_list_all = []  # haze features
    for i in range(block_rownum):
        for j in range(block_colnum):
            features1_list = []  # haze-free features
            features2_list = []  # haze features
            crop_row_start = i*BLOCK_SIZE_ROW
            crop_row_end = (i+1)*BLOCK_SIZE_ROW
            crop_col_start = j*BLOCK_SIZE_COL
            crop_col_end = (j+1)*BLOCK_SIZE_COL

            crop_gray = gray[crop_row_start: crop_row_end, crop_col_start:crop_col_end]
            crop_img = img[crop_row_start: crop_row_end, crop_col_start:crop_col_end]
            crop_gradient = gradient[crop_row_start: crop_row_end, crop_col_start:crop_col_end]
            crop_gradient2 = gradient2[crop_row_start: crop_row_end, crop_col_start:crop_col_end]
            crop_dark_image = dark_image[crop_row_start: crop_row_end, crop_col_start:crop_col_end]

            if np.mean(crop_dark_image) < DARK_CHANNEL_THRESHOLD_L:
                if np.count_nonzero(crop_gradient2) > 400:
                    # print('1', crop_gray.astype(np.float64))
                    features1_list.extend(compute_aggd.compute_features(crop_gray.astype(np.float64), crop_gradient.astype(np.float64)))
                    cv2.rectangle(img, (crop_col_start, crop_row_start), (crop_col_end, crop_row_end), (0, 255, 0))
                else:
                    # print('2', crop_gray.astype(np.float64))
                    features1_list.extend(compute_aggd.compute_features(crop_gray.astype(np.float64), crop_gradient.astype(np.float64)))
                    cv2.rectangle(img, (crop_col_start, crop_row_start), (crop_col_end, crop_row_end), (255, 0, 0))

            elif np.mean(crop_dark_image) >= DARK_CHANNEL_THRESHOLD_L:
                features2_list.extend(compute_aggd.compute_features(crop_gray.astype(np.float64), crop_gradient.astype(np.float64)))
                cv2.rectangle(img, (crop_col_start, crop_row_start), (crop_col_end, crop_row_end), (255, 0, 255))

            features1_list_all.extend(features1_list)
            features2_list_all.extend(features2_list)

    if len(features1_list_all) != 0:
        features1 = np.array(features1_list_all).reshape((int(len(features1_list_all)/FEATURE_NUMBER)), FEATURE_NUMBER)
        if features1.shape[0] >1:
            mu_distparam1 = (np.mean(features1, axis=0))
            cov_distparam1 = np.cov(features1.reshape(features1.shape[1], features1.shape[0]))
            invcov_param1 = np.linalg.inv((cov_prisparam1+cov_distparam1)/2)
            q1 = np.sqrt(np.dot(np.dot((mu_prisparam1-mu_distparam1),invcov_param1), np.transpose(mu_prisparam1-mu_distparam1)))
            quality.append(np.nanmean(q1))
        else:
            features2_list_all.extend(features2_list_all)

    if len(features2_list_all) != 0:
        features2 = np.array(features2_list_all).reshape((int(len(features2_list_all)/FEATURE_NUMBER)), FEATURE_NUMBER)
        #input(features2)
        mu_distparam2 = (np.mean(features2, axis=0))
        cov_distparam2 = np.cov(features2.reshape(features2.shape[1], features2.shape[0]))
        #input(mu_distparam2)
        invcov_param2 = np.linalg.inv((cov_prisparam2+cov_distparam2)/2)
        q2 = np.sqrt(np.dot(np.dot((mu_prisparam2-mu_distparam2),invcov_param2), np.transpose(mu_prisparam2-mu_distparam2)))
        # input(q2)
        quality.append(np.nanmean(q2))

    r = 0
    print(quality)
    for i in quality:
        r += i
    print('AV: %f' % r)


authenticity('../img/result.jpg')
