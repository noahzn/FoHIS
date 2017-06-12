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



BLOCK_SIZE_ROW     = 48
BLOCK_SIZE_COL     = 48
NORMALIZED_WIDTH   = 528
FEATURE_NUMBER     = 16
GRADIENT_THRESHOLD_L = 3
GRADIENT_THRESHOLD_R = 60
DARK_CHANNEL_THRESHOLD_L = 30
DARK_CHANNEL_THRESHOLD_R = 100



features1_list_all = []  # haze-free features
features2_list_all = []  # haze features
for root, dirs, files in os.walk('dataset/haze2'):
    for fn in files:
        img = cv2.resize(cv2.imread(os.path.join(root, fn)), (NORMALIZED_WIDTH, NORMALIZED_WIDTH), interpolation=cv2.INTER_CUBIC)
        for iters in range(1):
            print(fn, iters)
            img = cv2.resize(img, (int(NORMALIZED_WIDTH/(iters+1)), int(NORMALIZED_WIDTH/(iters+1))), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            BLOCK_SIZE_COL = int(BLOCK_SIZE_COL/(iters+1))
            BLOCK_SIZE_ROW = int(BLOCK_SIZE_ROW/(iters+1))

            block_rownum = math.floor(gray.shape[0]/BLOCK_SIZE_ROW)
            block_colnum = math.floor(gray.shape[1]/BLOCK_SIZE_COL)
            img = img[:block_rownum*BLOCK_SIZE_ROW, :block_colnum*BLOCK_SIZE_COL, :]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # gradient magnitude
            gradx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grady = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            absX = cv2.convertScaleAbs(gradx)   # 转换为uint8
            absY = cv2.convertScaleAbs(grady)
            gradient = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            gradient2 = gradient
            gradient2[gradient2 < 20] = 0
            gradient2[gradient2 >= 20] =255
            #cv2.imshow('g', gradient)
            #cv2.waitKey()


            # dark channel

            dark_image = np.asarray(guided_filter.getDark(Image.fromarray(np.uint8(img)), guided_filter.minimizeFilter
                                                          ,(int(10/(iters+1)), int(10/(iters+1)))))



            for i in range(block_rownum):
                for j in range(block_colnum):
                    features1_list = []  # haze-free features
                    features2_list = []  # haze features
                    crop_row_start = i*BLOCK_SIZE_ROW
                    crop_row_end = (i+1)*BLOCK_SIZE_ROW
                    crop_col_start = j*BLOCK_SIZE_COL
                    crop_col_end = (j+1)*BLOCK_SIZE_COL

                    crop_gray = gray[crop_row_start: crop_row_end, crop_col_start:crop_col_end]
                    crop_gradient = gradient[crop_row_start: crop_row_end, crop_col_start:crop_col_end]
                    crop_gradient2 = gradient2[crop_row_start: crop_row_end, crop_col_start:crop_col_end]
                    crop_dark_image = dark_image[crop_row_start: crop_row_end, crop_col_start:crop_col_end]

                    # print(crop_gray.shape)

                    if np.mean(crop_dark_image) < DARK_CHANNEL_THRESHOLD_L:
                        if np.count_nonzero(crop_gradient2) > 400:
                            features1_list.extend(compute_aggd.compute_features(crop_gray.astype(np.float64), crop_gradient.astype(np.float64)))
                            cv2.rectangle(img, (crop_col_start, crop_row_start), (crop_col_end, crop_row_end), (0, 255, 0))
                        else:
                            features1_list.extend(compute_aggd.compute_features(crop_gray.astype(np.float64), crop_gradient.astype(np.float64)))
                            cv2.rectangle(img, (crop_col_start, crop_row_start), (crop_col_end, crop_row_end), (255, 0, 0))

                    elif np.mean(crop_dark_image) > DARK_CHANNEL_THRESHOLD_R:
                        if np.count_nonzero(crop_gradient2) < 80:
                            features2_list.extend(compute_aggd.compute_features(crop_gray.astype(np.float64), crop_gradient.astype(np.float64)))
                            cv2.rectangle(img, (crop_col_start, crop_row_start), (crop_col_end, crop_row_end), (255, 0, 255))


                    features1_list_all.extend(features1_list)
                    features2_list_all.extend(features2_list)

            print(len(features1_list_all), len(features2_list_all))
            cv2.imshow('img', img)
            cv2.waitKey()

features1 = np.array(features1_list_all).reshape((int(len(features1_list_all)/FEATURE_NUMBER)), FEATURE_NUMBER)
features2 = np.array(features2_list_all).reshape((int(len(features2_list_all)/FEATURE_NUMBER)), FEATURE_NUMBER)
mu_prisparam1 = (np.mean(features1, axis=0))
mu_prisparam2 = (np.mean(features2, axis=0))

cov_prisparam1 = np.cov(features1.reshape(features1.shape[1], features1.shape[0]))
cov_prisparam2 = np.cov(features2.reshape(features2.shape[1], features2.shape[0]))

print(features2.shape, cov_prisparam2.shape)
sio.savemat('prisparam_16_hazeandfog.mat', {'mu1': mu_prisparam1, 'mu2': mu_prisparam2, 'cov1': cov_prisparam1, 'cov2': cov_prisparam2})