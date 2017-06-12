"""
Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity
Ning Zhang, Lin Zhang*, and Zaixi Cheng
"""

from PIL import Image
import numpy as np


def dot(matrix1, matrix2, operation):
    """dot operation for the matrix1 and matrix2"""
    out = []
    size = len(matrix1), len(matrix1[0])

    for x in range(size[0]):
        temp = []
        for y in range(size[1]):
            temp.append(operation(matrix1[x][y], matrix2[x][y]))

        out.append(temp)

    return out


def convertImageToMatrix(image):
    size = image.size
    out = []

    for x in range(size[1]):
        temp = []
        for y in range(size[0]):
            temp.append(image.getpixel((y, x)))

        out.append(temp)

    return out


def guidedFilter(srcImage, guidedImage, radius, epsilon):
    """guided filter for the image
     src image must be gray image
     guided image must be gray image
    """

    size = srcImage.size
    src = convertImageToMatrix(srcImage)
    guided = convertImageToMatrix(guidedImage)

    one = []

    for x in range(size[1]):
        one.append([1.0] * size[0])

    n = boxFilter(one, radius)

    plus = lambda x, y: x + y
    minus = lambda x, y: x - y
    multiple = lambda x, y: x * y
    divide = lambda x, y: x / y

    meanI = dot(boxFilter(src, radius), n, divide)
    meanP = dot(boxFilter(guided, radius), n, divide)
    meanIP = dot(boxFilter(dot(src, guided, multiple), radius), n, divide)

    covIP = dot(meanIP, dot(meanI, meanP, multiple), minus)

    meanII = dot(boxFilter(dot(src, src, multiple), radius), n, divide)
    varI = dot(meanII, dot(meanI, meanI, multiple), minus)

    epsilonMatrix = []

    for x in range(size[1]):
        epsilonMatrix.append([epsilon] * size[0])

    a = dot(covIP, dot(varI, epsilonMatrix, plus), divide)
    b = dot(meanP, dot(a, meanI, multiple), minus)

    meanA = dot(boxFilter(a, radius), n, divide)
    meanB = dot(boxFilter(b, radius), n, divide)

    return dot(dot(meanA, src, multiple), meanB, plus)


def boxFilter(im, radius):
    """box filter for the image of the radius"""
    height, width = len(im), len(im[0])

    imDst = []
    imCum = []

    for x in range(height):
        imDst.append([0.0] * width)
        imCum.append([0.0] * width)

    #cumulative sum over Y axis
    for i in range(width):
        for j in range(height):
            if j == 0:
                imCum[j][i] = im[j][i]
            else:
                imCum[j][i] = im[j][i] + imCum[j - 1][i]

    #difference over Y axis
    for j in range(radius + 1):
        for i in range(width):
            imDst[j][i] = imCum[j + radius][i]

    for j in range(radius + 1, height - radius):
        for i in range(width):
            imDst[j][i] = imCum[j + radius][i] - imCum[j - radius - 1][i]

    for j in range(height - radius, height):
        for i in range(width):
            imDst[j][i] = imCum[height - 1][i] - imCum[j - radius - 1][i]

    #cumulative sum over X axis
    for j in range(height):
        for i in range(width):
            if i == 0:
                imCum[j][i] = imDst[j][i]
            else:
                imCum[j][i] = imDst[j][i] + imCum[j][i - 1]

    #difference over X axis
    for j in range(height):
        for i in range(radius + 1):
            imDst[j][i] = imCum[j][i + radius]

    for j in range(height):
        for i in range(radius + 1, width - radius):
            imDst[j][i] = imCum[j][i + radius] - imCum[j][i - radius - 1]

    for j in range(height):
        for i in range(width - radius, width):
            imDst[j][i] = imCum[j][width - 1] - imCum[j][i - radius - 1]

    return imDst


def getLight(srcImage, darkImage, cut):
    """get atmospheric light from the picture"""
    size = darkImage.size
    light = []

    for x in range(size[0]):
        for y in range(size[1]):
            light.append(darkImage.getpixel((x, y)))

    light.sort()
    light.reverse()

    threshold = light[int(cut * len(light))]

    atmosphere = {}

    for x in range(size[0]):
        for y in range(size[1]):
            if darkImage.getpixel((x, y)) >= threshold:
                atmosphere.update({(x, y): sum(srcImage.getpixel((x, y))) / 3.0})

    pos = sorted(atmosphere.items(), key = lambda item: item[1], reverse = True)[0][0]

    return srcImage.getpixel(pos)


def getTransmission(input_img, light, omiga):
    """get transmission from the picture"""
    size = input_img.size
    output = []

    for x in range(size[1]):
        temp = []
        for y in range(size[0]):
            temp.append(min(input_img.getpixel((y, x))) / float(min(light)))

        output.append(temp)

    transmission = []

    for x in range(size[1]):
        temp = []
        for y in range(size[0]):
            temp.append(1 - omiga * minimizeFilter(output, (x, y), (10, 10)))

        transmission.append(temp)

    return transmission


def minimizeFilter(input_img, point, size):
    """minimize filter for the input image"""
    begin = (point[0] - size[0] / 2, point[0] + size[0] / 2 + 1)
    end = (point[1] - size[1] / 2, point[1] + size[1] / 2 + 1)

    begin1, begin2 = int(begin[0]), int(begin[1])
    end1, end2 = int(end[0]), int(end[1])

    l = []

    for i in range(begin1, begin2):
        for j in range(end1, end2):
            if (i >= 0 and i < len(input_img)) and (j >= 0 and j < len(input_img[0])):
                l.append(input_img[i][j])

    return min(l)


def ensure(n):
    if n < 0:
        n = 0

    if n > 255:
        n = 255

    return int(n)


def guided_filter_image(img):
    dark = np.asarray(img).min(axis=2)
    light = getLight(img, Image.fromarray(dark), 0.001)

    transmission = getTransmission(img, light, 0.9)

    tranImage = Image.new('L', img.size)
    grayImage = img.convert('L')

    for x in range(img.size[0]):
        for y in range(img.size[1]):
            tranImage.putpixel((x, y), int(transmission[y][x] * 255))


    guided = guidedFilter(grayImage, tranImage, 25, 0.001)

    guidedImage = Image.new('L', img.size)

    for x in range(img.size[0]):
        for y in range(img.size[1]):
            guidedImage.putpixel((x, y), ensure(guided[y][x]))


    return guidedImage


def getDark(input_img, filter, frame):
    """get dark image from the input image"""
    size = input_img.size
    output = []

    for x in range(size[1]):
        temp = []
        for y in range(size[0]):
            temp.append(min(input_img.getpixel((y, x))))

        output.append(temp)

    output = filter2d(output, filter, frame)

    output_img = Image.new('L', size)

    for x in range(size[1]):
        for y in range(size[0]):
            output_img.putpixel((y, x), output[x][y])

    return output_img


def filter2d(input_img, filter, frame):
    """filter of the 2-dimension picture"""
    size = len(input_img), len(input_img[0])
    output = []

    for i in range(size[0]):
        temp = []
        for j in range(size[1]):
            temp.append(filter(input_img, (i, j), frame))

        output.append(temp)

    return output