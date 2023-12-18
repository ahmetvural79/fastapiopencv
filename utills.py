#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import cv2
import math
import numpy as np
import os.path
from PIL import Image
import io
import json

import os


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def remove_isolated_pixels(image):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] == 1:
            new_image[labels == label] = 0

    return new_image


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def remove_isolated_pixels(image):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] == 1:
            new_image[labels == label] = 0

    return new_image
 
#
def read_cv2_image(binaryimg):

    stream = io.BytesIO(binaryimg)

    image = np.asarray(bytearray(stream.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

def opResize(binimage,w=None,h=None,l=None):
    data = {"success": False}
    if binimage is None:
        return data
      # convert the binary image to image
    image = read_cv2_image(binimage)

    #filename=os.path.basename(filnm)

    [iHeight, iWidth, channels]=image.shape
    oWidth=128
    oHeight=0
    _lambda=1.0
    if w is not None:
        oWidth = w
    if h is not None:
        oHeight =h
    if l is not None:
        _lambda = l



    if oWidth  == 0:
        oWidth  = round(iWidth  * oHeight/iHeight)
    if oHeight == 0:
        oHeight = round(iHeight * oWidth /iWidth)
    #outputFilename = str(filename+'_'+str(oWidth)+'x'+str(oHeight)+'_'+str(_lambda)+'.png')
    print(oWidth,oHeight,_lambda)
    avgImage = np.zeros([oHeight, oWidth, channels])
    oImage   = np.zeros([oHeight, oWidth, channels])

    pWidth  = iWidth  / oWidth
    pHeight = iHeight / oHeight
    #calc average image
    for py in range(oHeight):
        for px in range(oWidth):
            sx = max(px * pWidth, 0)
            ex = min((px+1) * pWidth, iWidth)
            sy = max(py * pHeight, 0)
            ey = min((py+1) * pHeight, iHeight)

            sxr = math.floor(sx)
            syr = math.floor(sy)
            exr = math.ceil(ex)
            eyr = math.ceil(ey)

            avgF = 0
        
            for iy in range(syr,eyr):
                for ix in range(sxr,exr):
                    f=1
                    if(ix < sx):
                        f = f * (1.0 - (sx - ix))
                    if((ix+1) > ex):
                        f = f * (1.0 - ((ix+1) - ex))
                    if(iy < sy):
                        f = f * (1.0 - (sy - iy))
                    if((iy+1) > ey):
                        f = f * (1.0 - ((iy+1) - ey))
                    avgImage[py, px, :] = avgImage[py, px, :] + (image[iy, ix, :] * f)
                    avgF = avgF + f
            avgImage[py, px, :] = avgImage[py, px, :] / avgF
    #cv2.imwrite("avg.png", avgImage)
    #calc output image
    for py in range(oHeight):
        for px in range(oWidth):
            avg=np.zeros([1, channels + 1])
            if(py > 0):
                if(px > 0):
                    avg = avg + np.append(np.reshape(avgImage[py-1, px-1,   :], [1,channels]) * 1,1)
                avg = avg + np.append(np.reshape(avgImage[py-1, px+0, :], [1,channels]) * 2,2)
                if((px+1) < oWidth):
                    avg = avg + np.append(np.reshape(avgImage[py-1, px+1, :], [1,channels]) * 1,1)
            if(px > 0):
                avg = avg + np.append(np.reshape(avgImage[py+0, px-1,   :], [1,channels]) * 2,2)
            avg = avg + np.append(np.reshape(avgImage[py+0, px+0, :], [1,channels]) * 4,4)
            if((px+1) < oWidth):
                avg = avg + np.append(np.reshape(avgImage[py+0, px+1, :], [1,channels]) * 2,2)

            if((py+1) < oHeight):
                if(px > 0):
                    avg = avg + np.append(np.reshape(avgImage[py+1, px-1,   :], [1,channels]) * 1,1)
                avg = avg + np.append(np.reshape(avgImage[py+1, px+0, :], [1,channels]) * 2,2)
                if((px+1) < oWidth):
                    avg = avg + np.append(np.reshape(avgImage[py+1, px+1, :], [1,channels]) * 1,1)           
            if avg[0][3]==4:
                print(avg[0][3])
            avg = avg / avg[0][3]
            avg = avg[0][0:channels]
            sx = max(px * pWidth, 0)
            ex = min((px+1) * pWidth, iWidth)
            sy = max(py * pHeight, 0)
            ey = min((py+1) * pHeight, iHeight)

            sxr = math.floor(sx)
            syr = math.floor(sy)
            exr = math.ceil(ex)
            eyr = math.ceil(ey)

            oF = 0

            for iy in range(syr,(eyr)):
                for ix in range(sxr,(exr)):
                    if _lambda == 0:
                        f = 1
                    else:
                        f=np.linalg.norm(avg - np.reshape(image[iy + 0, ix + 0, :], [1,channels]),2)
                        #f=f/441.6729559
                        f = f **_lambda
                    
                    if(ix < sx):
                        f = f * (1.0 - (sx - ix))
                    if((ix+1) > ex):
                        f = f * (1.0 - ((ix+1) - ex))
                    if(iy < sy):
                        f = f * (1.0 - (sy - iy))
                    if((iy+1) > ey):
                        f = f * (1.0 - ((iy+1) - ey))
                    
                    oImage[py + 0, px + 0, :] = oImage[py + 0, px + 0, :] + (image[iy + 0, ix + 0, :] * f)
                    oF = oF + f

            if (oF == 0):
                oImage[py + 0, px + 0, :] = avg
            else:
                oImage[py + 0, px + 0, :] = oImage[py + 0, px + 0, :] / oF
               
    #cv2.imwrite(outputFilename, oImage)


    sharpened_image = unsharp_mask(oImage)
    #cv2.imwrite(outputFilename2, sharpened_image)
    #sharpened_image2 = unsharp_mask(sharpened_image)
    #cv2.imwrite(outputFilename3, sharpened_image2)
    return sharpened_image

