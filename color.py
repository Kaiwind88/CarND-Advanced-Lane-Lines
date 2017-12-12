import numpy as np
import cv2
from math import *
import matplotlib.pyplot as plt
from enum import Enum
import pickle
import os.path
import glob
from preprocess import *


images = glob.glob('./Capture7/*.jpg')
img = plt.imread(images[0])
mask = np.zeros(img.shape[:2], np.uint8)
print(img.shape)
width = img.shape[1]
height = img.shape[0]
v_bottom_left = (int(width * 0.16), int(height * 0.95))
v_bottom_right = (int(width * 0.94), int(height * 0.95))
v_top_left = (int(width * 0.47), int(height * 0.59))
v_top_right = (int(width * 0.58), int(height * 0.59))
src = np.float32([[579, 460], [704, 460], [1110, 719.5], [200, 719.5]])
dst = np.float32([[320, 0.5], [960, 0.5], [960, 719.5], [320, 719.5]])

vertices = np.array([[v_bottom_left, v_top_left, v_top_right, v_bottom_right]], dtype=np.int32)
cv2.fillPoly(mask, vertices, 255)
# mask[100:300, 100:400] = 255

def draw1():
    for img in images:
        images_i = plt.imread(img)

        gray = cv2.cvtColor(images_i, cv2.COLOR_RGB2GRAY)
        equ = cv2.equalizeHist(gray)
        color_equ =  cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)

        half = images_i.shape[0] // 2
        images_i = images_i[half:, :, :]
        color = cv2.cvtColor(images_i, cv2.COLOR_RGB2LUV)
        gray1 = cv2.cvtColor(images_i, cv2.COLOR_RGB2GRAY)
        B = images_i[:,:,2]
        S = color[:,:,2]
        B = cv2.equalizeHist(S)
        zero = np.zeros_like(S)
        gray = (B+S+gray1)/3#np.dstack((B, S, zero))
        plt.imshow(color_equ[:,:,2], cmap='gray')
        # plt.show()
        # # ret, th = cv2.threshold(color[:,:,2], 200, 255,type=0)
        # # print(ret)
        # masked_img = cv2.bitwise_and(color, color, mask=mask)
        # # Calculate histogram with mask and without mask
        # # Check third argument for mask
        # # hist_full = cv2.calcHist([color], [0], None, [1280], [0, 1280])
        # # hist_mask = cv2.calcHist([color], [0], mask, [1280], [0, 1280])
        # histogram = np.sum(masked_img[masked_img.shape[0] // 2:, :, 2], axis=0)

        # plt.subplot(221), plt.imshow(B, 'gray')
        # plt.subplot(222), plt.imshow(S, 'gray')
        # plt.subplot(223), plt.imshow(gray1, 'gray')
        # plt.subplot(224), plt.imshow(gray, 'gray')

        # plt.subplot(224),  plt.plot(histogram)
        # plt.xlim([0, 1280])
        plt.show()

def draw2(img):
    # create a mask
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[450:, :] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    # Calculate histogram with mask and without mask
    # Check third argument for mask
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(224), plt.plot(hist_full, 'red'), plt.plot(hist_mask)
    plt.xlim([0, 256])
    plt.show()

def contrast(gray):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
    res = cv2.normalize(close, close, 0, 255, cv2.NORM_MINMAX)
    return res

def otsu(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plot all the images and their histograms
    return th

load_data()
mtx = parameters['mtx']
dist = parameters['dist']
M = parameters['M']
MInv = parameters['MInv']
nx = 9
ny = 6
for img in images[::30]:
    image = plt.imread(img)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    M, MInv = transform(image, nx, ny, mtx, dist)
    unwarp_img = unwarp(image,M,mtx,dist)
    s = hls[:,:,2]
    # otsu(s)
    # contrast = contrast(gray)
    plt.subplot(221), plt.imshow(image)
    plt.subplot(222), plt.imshow(unwarp_img)
    # plt.imshow(s, cmap='gray')
    plt.show()
    # draw2(contrast)