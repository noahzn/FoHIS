"""
Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity
Ning Zhang, Lin Zhang*, and Zaixi Cheng
"""

import cv2
import numpy as np
import tool_kit as tk
from parameter import const

np.set_printoptions(threshold=np.inf)
np.errstate(invalid='ignore', divide='ignore')

# load rgb and depth image
img = '../img/img.jpg'
Ip = cv2.imread(img)
# name = str(img.split('.')[0].split('/')[1])
depth = cv2.imread('../img/imgd.jpg')[:, :, 0].astype(np.float64)
depth[depth==0] = 1  # the depth_min shouldn't be 0
depth *= 3

I = np.empty_like(Ip)
result = np.empty_like(Ip)

elevation, distance, angle = tk.elevation_and_distance_estimation(img, depth,
                                                         const.CAMERA_VERTICAL_FOV,
                                                         const.HORIZONTAL_ANGLE,
                                                         const.CAMERA_ALTITUDE)


if const.FT != 0:
    perlin = tk.noise(Ip, depth)
    ECA = const.ECA
    # ECA = const.ECA * np.exp(-elevation/(const.FT+0.00001))
    c = (1-elevation/(const.FT+0.00001))
    c[c<0] = 0

    if const.FT > const.HT:
        ECM = (const.ECM * c + (1-c)*const.ECA) * (perlin/255)
    else:
        ECM = (const.ECA * c + (1-c)*const.ECM) * (perlin/255)

else:
    ECA = const.ECA
    # ECA = const.ECA * np.exp(-elevation/(const.FT+0.00001))
    ECM = const.ECM


distance_through_fog = np.zeros_like(distance)
distance_through_haze = np.zeros_like(distance)
distance_through_haze_free = np.zeros_like(distance)


if const.FT == 0:  # only haze: const.FT should be set to 0
    idx1 = elevation > const.HT
    idx2 = elevation <= const.HT

    if const.CAMERA_ALTITUDE <= const.HT:
        distance_through_haze[idx2] = distance[idx2]
        distance_through_haze_free[idx1] = (elevation[idx1] - const.HT) * distance[idx1] \
                                          / (elevation[idx1] - const.CAMERA_ALTITUDE)

        distance_through_haze[idx1] = distance[idx1] - distance_through_haze_free[idx1]

    elif const.CAMERA_ALTITUDE > const.HT:
        distance_through_haze_free[idx1] = distance[idx1]
        distance_through_haze[idx2] = (const.HT - elevation[idx2]) * distance[idx2] \
                                     / (const.CAMERA_ALTITUDE - elevation[idx2])
        distance_through_haze_free[idx2] = distance[idx2] - distance_through_fog[idx2]

    I[:, :, 0] = Ip[:, :, 0] * np.exp(-ECA*distance_through_haze-ECM*distance_through_haze_free)
    I[:, :, 1] = Ip[:, :, 1] * np.exp(-ECA*distance_through_haze-ECM*distance_through_haze_free)
    I[:, :, 2] = Ip[:, :, 2] * np.exp(-ECA*distance_through_haze-ECM*distance_through_haze_free)
    O = 1-np.exp(-ECA*distance_through_haze-const.ECM*distance_through_haze_free)


elif const.FT < const.HT and const.FT != 0:
    idx1 = (np.logical_and(const.HT > elevation, elevation > const.FT))
    idx2 = elevation <= const.FT
    idx3 = elevation >= const.HT
    if const.CAMERA_ALTITUDE <= const.FT:
        distance_through_fog[idx2] = distance[idx2]
        distance_through_haze[idx1] = (elevation[idx1] - const.FT) * distance[idx1] \
                                          / (elevation[idx1] - const.CAMERA_ALTITUDE)

        distance_through_fog[idx1] = distance[idx1] - distance_through_haze[idx1]
        distance_through_fog[idx3] = (const.FT - const.CAMERA_ALTITUDE) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)
        distance_through_haze[idx3] = (const.HT - const.FT) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)
        distance_through_haze_free[idx3] = distance[idx3] - distance_through_haze[idx3] - distance_through_fog[idx3]

    elif const.CAMERA_ALTITUDE > const.HT:
        distance_through_haze_free[idx3] = distance[idx3]
        distance_through_haze[idx1] = (const.FT - elevation[idx1]) * distance_through_haze_free[idx1] \
                                     / (const.CAMERA_ALTITUDE - const.HT)
        distance_through_haze_free[idx1] = distance[idx1] - distance_through_haze[idx1]


        distance_through_fog[idx2] = (const.FT - elevation[idx2]) * distance[idx2] / (const.CAMERA_ALTITUDE - elevation[idx2])
        distance_through_haze[idx2] = (const.HT - const.FT) * distance / (const.CAMERA_ALTITUDE - elevation[idx2])
        distance_through_haze_free[idx2] = distance[idx2] - distance_through_haze[idx2] - distance_through_fog[idx2]

    elif const.FT < const.CAMERA_ALTITUDE <= const.HT:
        distance_through_haze[idx1] = distance[idx1]
        distance_through_fog[idx2] = (const.FT - elevation[idx2]) * distance[idx2] / (const.CAMERA_ALTITUDE - elevation[idx2])
        distance_through_haze[idx2] = distance[idx2] - distance_through_fog[idx2]
        distance_through_haze_free[idx3] = (elevation[idx3] - const.HT) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)
        distance_through_haze[idx3] = (const.HT - const.CAMERA_ALTITUDE) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)

    I[:, :, 0] = Ip[:, :, 0] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
    I[:, :, 1] = Ip[:, :, 1] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
    I[:, :, 2] = Ip[:, :, 2] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
    O = 1-np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)


elif const.FT > const.HT:
    if const.CAMERA_ALTITUDE <= const.HT:
        idx1 = (np.logical_and(const.FT > elevation, elevation > const.HT))
        idx2 = elevation <= const.HT
        idx3 = elevation >= const.FT

        distance_through_haze[idx2] = distance[idx2]
        distance_through_fog[idx1] = (elevation[idx1] - const.HT) * distance[idx1] \
                                          / (elevation[idx1] - const.CAMERA_ALTITUDE)
        distance_through_haze[idx1] = distance[idx1] - distance_through_fog[idx1]
        distance_through_haze[idx3] = (const.HT - const.CAMERA_ALTITUDE) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)
        distance_through_fog[idx3] = (const.FT - const.HT) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)
        distance_through_haze_free[idx3] = distance[idx3] - distance_through_haze[idx3] - distance_through_fog[idx3]

    elif const.CAMERA_ALTITUDE > const.FT:
        idx1 = (np.logical_and(const.HT > elevation, elevation > const.FT))
        idx2 = elevation <= const.FT
        idx3 = elevation >= const.HT

        distance_through_haze_free[idx3] = distance[idx3]
        distance_through_haze[idx1] = (const.FT - elevation[idx1]) * distance_through_haze_free[idx1] \
                                     / (const.CAMERA_ALTITUDE - const.HT)
        distance_through_haze_free[idx1] = distance[idx1] - distance_through_haze[idx1]
        distance_through_fog[idx2] = (const.FT - elevation[idx2]) * distance[idx2] / (const.CAMERA_ALTITUDE - elevation[idx2])
        distance_through_haze[idx2] = (const.HT - const.FT) * distance / (const.CAMERA_ALTITUDE - elevation[idx2])
        distance_through_haze_free[idx2] = distance[idx2] - distance_through_haze[idx2] - distance_through_fog[idx2]

    elif const.HT < const.CAMERA_ALTITUDE <= const.FT:
        idx1 = (np.logical_and(const.HT > elevation, elevation > const.FT))
        idx2 = elevation <= const.FT
        idx3 = elevation >= const.HT

        distance_through_haze[idx1] = distance[idx1]
        distance_through_fog[idx2] = (const.FT - elevation[idx2]) * distance[idx2] / (const.CAMERA_ALTITUDE - elevation[idx2])
        distance_through_haze[idx2] = distance[idx2] - distance_through_fog[idx2]
        distance_through_haze_free[idx3] = (elevation[idx3] - const.HT) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)
        distance_through_haze[idx3] = (const.HT - const.CAMERA_ALTITUDE) * distance[idx3] / (elevation[idx3] - const.CAMERA_ALTITUDE)

    I[:, :, 0] = Ip[:, :, 0] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
    I[:, :, 1] = Ip[:, :, 1] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
    I[:, :, 2] = Ip[:, :, 2] * np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)
    O = 1-np.exp(-ECA*distance_through_haze-ECM*distance_through_fog)

Ial = np.empty_like(Ip)  # color of the fog/haze
Ial[:, :, 0] = 225
Ial[:, :, 1] = 225
Ial[:, :, 2] = 201
# Ial[:, :, 0] = 240
# Ial[:, :, 1] = 240
# Ial[:, :, 2] = 240

result[:, :, 0] = I[:, :, 0] + O * Ial[:, :, 0]
result[:, :, 1] = I[:, :, 1] + O * Ial[:, :, 1]
result[:, :, 2] = I[:, :, 2] + O * Ial[:, :, 2]

cv2.imwrite('../img/result.jpg', result)
