import numpy as np
import cv2

from math import *
import matplotlib.pyplot as plt
import pickle
import os.path
from enum import Enum
from find_lane import *
from preprocess import *

load_data()
mtx = parameters['mtx']
dist = parameters['dist']
M = parameters['M']
MInv = parameters['MInv']
img = cv2.imread('./test_images/straight_lines1.jpg')
# # img = cv2.imread('./camera_cal/calibration8.jpg')
# M, MInv = transform(img, nx, ny, mtx, dist)
# unwarp_img = unwarp(img, M, mtx, dist)
# undist = cal_undistort(img, mtx, dist)
# show_images(undist, unwarp_img)
# warp_img = unwarp(unwarp_img, MInv, mtx, dist)
# show_images(undist, warp_img)


combined, undist_img, single_channel_img, hls_img, rgb_img, xsobel, ysobel, msobel, dsobel = pipline(img, parameters)
image_dict = {
    1: ['undist', undist_img],
    2: ['s_channel', single_channel_img],
    3: ['xsobel', xsobel],
    4: ['ysobel', ysobel],
    5: ['msobel', msobel],
    6: ['dsobel', dsobel],
    7: ['combined', combined],
}

# show_processed_images(image_dict)


w_name = 'image'
cv2.namedWindow(w_name, cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture('project_video.mp4')
# cap = cv2.VideoCapture('challenge_video.mp4')
# cap = cv2.VideoCapture('harder_challenge_video.mp4')

def nothing(x):
    cap.set(cv2.CAP_PROP_POS_FRAMES, x)
cv2.createTrackbar('Frame',w_name,0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),nothing)

while(cap.isOpened()):
    if key_handler(1, parameters):
        break

    if parameters['p']:
        ret, frame = cap.read()
        cv2.setTrackbarPos('Frame',w_name, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if not ret:
            continue

    orig = np.copy(frame)
    # gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # equ_img = clahe.apply(gray)
    # orig = cv2.cvtColor(equ_img, cv2.COLOR_GRAY2BGR)


    combined, undist_img, s, hls_img, rgb_img, _, _, _, _= pipline(orig, parameters)

    scale = 0.7
    resize_shape = (int(orig.shape[1]*scale), int(orig.shape[0]*scale))
    unwarped = unwarp(combined, M, mtx, dist)
    left_fit, right_fit = find_lane(unwarped)
    stacked_img, left_curverad, right_curverad, left_fitx, right_fitx= find_lane_skip_window(unwarped, left_fit, right_fit)
    parameters['left_curverad'] = left_curverad
    parameters['right_curverad'] = right_curverad
    # stacked_img = find_lane_cnn(image)
    show_text(orig, parameters)
    # stacked_img = np.dstack((rgb_img, hls_img, combined))*255
    # stacked_img = np.dstack((image, image, image)) * 255
    if stacked_img is None or left_fitx is None or right_fitx is None:
        continue
    resize_img = cv2.resize(stacked_img, resize_shape, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('unwarp', resize_img)
    resize_img = cv2.resize(orig, resize_shape, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(w_name, resize_img)
    project = project_back(undist_img, unwarped, MInv, left_fitx, right_fitx)
    resize_img = cv2.resize(project, resize_shape, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('project', resize_img)

cap.release()
cv2.destroyAllWindows()
