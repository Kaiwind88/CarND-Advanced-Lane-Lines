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

M_max = parameters['M_max']
MInv_max = parameters['MInv_max']
M_mid = parameters['M_mid']
MInv_mid = parameters['MInv_mid']
M_min = parameters['M_min']
MInv_min = parameters['MInv_min']

M = M_max
MInv = MInv_max
parameters['p'] = True
# img = cv2.imread('./test_images/straight_lines1.jpg')
# # img = cv2.imread('./camera_cal/calibration8.jpg')
# M, MInv = transform(img, nx, ny, mtx, dist)
# unwarp_img = unwarp(img, M, mtx, dist)
# undist = cal_undistort(img, mtx, dist)
# show_images(undist, unwarp_img)
# warp_img = unwarp(unwarp_img, MInv, mtx, dist)
# show_images(undist, warp_img)


# combined, undist_img, single_channel_img, hls_img, rgb_img, xsobel, ysobel, msobel, dsobel = pipline(img, parameters)
# image_dict = {
#     1: ['undist', undist_img],
#     2: ['s_channel', single_channel_img],
#     3: ['xsobel', xsobel],
#     4: ['ysobel', ysobel],
#     5: ['msobel', msobel],
#     6: ['dsobel', dsobel],
#     7: ['combined', combined],
# }

# show_processed_images(image_dict)

f1 = 'project_video.mp4'
f2 = 'challenge_video.mp4'
f3 = 'harder_challenge_video.mp4'
input_file = f3
if input_file == f3:
    parameters['M'] = M_mid
    parameters['MInv'] = MInv_mid
else:
    parameters['M'] = M_max
    parameters['MInv'] = MInv_max

lane = Lane(parameters)
w_name = input_file
cv2.namedWindow(w_name, cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(input_file)

def progress_bar_cb(x):
    cap.set(cv2.CAP_PROP_POS_FRAMES, x)
cv2.createTrackbar('Frame',w_name,0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),progress_bar_cb)

delay = 1
while(cap.isOpened()):
    if key_handler(delay, parameters):
        break
    if parameters['s']:
        delay = 0
    else:
        delay = 1

    if parameters['p']:
        ret, frame = cap.read()
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos('Frame',w_name, frame_idx)
        set_frame_idx(frame_idx)
        parameters['frame'] = frame_idx
        if not ret:
            continue
    print("--------current frame:", frame_idx)

    final_img = lane.lane_detection(frame)
    cv2.imshow(w_name, final_img)

cap.release()
cv2.destroyAllWindows()

