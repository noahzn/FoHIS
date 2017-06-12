"""
Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity
Ning Zhang, Lin Zhang*, and Zaixi Cheng
"""

import cv2
import numpy as np
from scipy.special import gamma
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)


def compute_features(gray, gradient):
    features = []

    for m in [gray]:
        # MSCN
        μ = cv2.GaussianBlur(m, (5, 5), 5/6, borderType=cv2.BORDER_REPLICATE)
        σ = np.sqrt(abs(cv2.GaussianBlur(m*m, (5, 5), 5/6, borderType=cv2.BORDER_REPLICATE) - μ*μ))
        I = (m - μ) / (σ + 1)
        # print(I)


        # MSCN AGGD parameter
        # print(I)
        alpha, beta_l, beta_r = estimate_aggd_parameter(I)
        features.append(alpha)
        features.append((beta_l+beta_r)/2)

        # Log-Derivatives
        I = np.log(I + 0.00001)
        shift1 = [(0, 1), (1, 0), (1, 1), (1, -1)]
        shift2 = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (1, 1), (0, 1), (1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]

        for i in shift1:
            D = np.roll(I, i, axis=(0, 1)) - I
            alpha, beta_l, beta_r = estimate_aggd_parameter(D)
            features.append(alpha)
            features.append((beta_l+beta_r)/2)

        for i in range(3):
            D = np.roll(I, shift2[4*i], axis=(0, 1)) + np.roll(I, shift2[4*i+1], axis=(0, 1)) \
                - np.roll(I, shift2[4*i+2], axis=(0, 1)) - np.roll(I, shift2[4*i+3], axis=(0, 1))
            alpha, beta_l, beta_r = estimate_aggd_parameter(D)

            features.append(alpha)
            features.append((beta_l+beta_r)/2)

    return features


def estimate_aggd_parameter(vec):
    vec = np.nan_to_num(vec)
    # print(vec)
    gam = [x/1000 for x in range(200, 10001, 1)]
    r_gam = [(gamma(2/x)**2/(gamma(1/x)*gamma(3/x))) for x in gam]

    leftstd = np.nan_to_num(np.sqrt(np.mean(vec[vec < 0]**2)))

    rightstd = np.nan_to_num(np.sqrt(np.mean(vec[vec > 0]**2)))

    gammahat = np.nan_to_num(leftstd / (rightstd+0.00001))

    # print(leftstd, rightstd, gammahat)

    rhat = np.nan_to_num((np.mean(np.abs(vec))**2) / np.nanmean(vec**2))
    rhatnorm = (rhat*(gammahat**3 + 1)*(gammahat + 1))/((gammahat**2 + 1)**2)

    m1 = (r_gam - rhatnorm)**2

    m2 = m1.tolist()
    # print(r_gam)
    array_position = m2.index(np.min(m1))

    alpha = gam[array_position]
    beta_l = leftstd * np.sqrt(gamma(1/alpha)/gamma(3/alpha))
    beta_r = rightstd * np.sqrt(gamma(1/alpha)/gamma(3/alpha))

    return alpha, beta_l, beta_r
