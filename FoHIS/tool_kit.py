"""
Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity
Ning Zhang, Lin Zhang*, and Zaixi Cheng
"""

import numpy as np
import scipy.io as sio
import math
import cv2
from PIL import Image
from noise import pnoise3


# np.set_printoptions(threshold=np.inf)
np.errstate(invalid='ignore', divide='ignore')


def elevation_and_distance_estimation(src, depth, vertical_fov, horizontal_angle, camera_altitude):
    """
    estimate each pixel's elevation and distance
    :param src: RGB image
    :param depth: depth matrix
    :param vertical_fov: the camera vertical fov
    :param horizontal_angle: the angle between camera and horizontal line
    :param camera_altitude:
    :return: elevation and distance
    """

    img = cv2.imread(src)
    img_dpi = get_image_info(src)
    height, width = img.shape[:2]
    altitude = np.empty((height, width))
    distance = np.empty((height, width))
    angle = np.empty((height, width))
    depth_min = depth.min()

    for j in range(width):
        for i in range(height):
            theta = i / (height - 1) * vertical_fov

            horizontal_angle = 0  # the horizontal angles of images in our paper are all 0бу

            if horizontal_angle == 0:
                if theta < 0.5 * vertical_fov:
                    distance[i, j] = depth[i, j] / math.cos(math.radians(0.5 * vertical_fov - theta))
                    h_half = math.tan(0.5*vertical_fov)*depth_min
                    y2 = (0.5*height-i)/img_dpi[0]*2.56
                    y1 = h_half*y2/(height/img_dpi[0]*2.56)

                    altitude[i, j] = camera_altitude+depth[i, j]*y1/depth_min
                    angle[i, j] = 0.5 * vertical_fov - theta
                elif theta == 0.5 * vertical_fov:
                    distance[i, j] = depth[i, j]
                    h_half = math.tan(0.5*vertical_fov)*depth_min
                    y2 = (i-0.5*height)/img_dpi[0]*2.56
                    y1 = h_half*y2/(height/img_dpi[0]*2.56)

                    altitude[i, j] = max(camera_altitude - depth[i, j]*y1/depth_min, 0)
                    angle[i, j] = 0
                elif theta > 0.5 * vertical_fov:
                    distance[i, j] = depth[i, j] / math.cos(math.radians(theta-0.5*vertical_fov))
                    h_half = math.tan(0.5*vertical_fov)*depth_min
                    y2 = (i-0.5*height)/img_dpi[0]*2.56
                    y1 = h_half*y2/(height/img_dpi[0]*2.56)

                    altitude[i, j] = max(camera_altitude - depth[i, j]*y1/depth_min, 0)
                    angle[i, j] = -(theta - 0.5 * vertical_fov)

    return altitude, distance, angle


def get_image_info(src):
    im = Image.open(src)

    return im.info['dpi']

def noise(Ip, depth):
    p1 = Image.new('L', (Ip.shape[1], Ip.shape[0]))
    p2 = Image.new('L', (Ip.shape[1], Ip.shape[0]))
    p3 = Image.new('L', (Ip.shape[1], Ip.shape[0]))

    scale = 1/130.0
    for y in range(Ip.shape[0]):
        for x in range(Ip.shape[1]):
            v = pnoise3(x * scale, y * scale, depth[y, x] * scale, octaves=1, persistence=0.5, lacunarity=2.0)
            color = int((v+1)*128.0)
            p1.putpixel((x, y), color)

    scale = 1/60.0
    for y in range(Ip.shape[0]):
        for x in range(Ip.shape[1]):
            v = pnoise3(x * scale, y * scale, depth[y, x] * scale, octaves=1, persistence=0.5, lacunarity=2.0)
            color = int((v+0.5)*128)
            p2.putpixel((x, y), color)

    scale = 1/10.0
    for y in range(Ip.shape[0]):
        for x in range(Ip.shape[1]):
            v = pnoise3(x * scale, y * scale, depth[y, x] * scale, octaves=1, persistence=0.5, lacunarity=2.0)
            color = int((v+1.2)*128)
            p3.putpixel((x, y), color)

    perlin = (np.array(p1) + np.array(p2)/2 + np.array(p3)/4)/3

    return perlin