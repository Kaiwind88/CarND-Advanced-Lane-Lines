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


lane = Lane(MInv)

w_name = 'image'
cv2.namedWindow(w_name, cv2.WINDOW_AUTOSIZE)
# cap = cv2.VideoCapture('project_video.mp4')
# cap = cv2.VideoCapture('challenge_video.mp4')
cap = cv2.VideoCapture('harder_challenge_video.mp4')
# cap = cv2.VideoCapture('saveVideo.avi')

def nothing(x):
    cap.set(cv2.CAP_PROP_POS_FRAMES, x)
cv2.createTrackbar('Frame',w_name,0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),nothing)

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
    orig = frame
    # gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # equ_img = clahe.apply(gray)
    # orig = cv2.cvtColor(equ_img, cv2.COLOR_GRAY2BGR)
    # print(orig.shape)


    undist_img, wy_img, gray, color, combined, unwarp_img, edge, undist_img_enhance= pipline(orig, parameters)
    scale = 0.3

    resize_shape = (int(orig.shape[1]*scale), int(orig.shape[0]*scale))

    # print("unwarped", unwarped.shape, combined.shape)
    fit_lane_img = lane.fit_lane(unwarp_img)
    # fit_lane_img = find_lane_cnn(unwarp_img)
    # unwarped_img = unwarped
    offset, left_bestx, right_bestx, left_curverad, right_curverad = lane.get_lane_prop()

    parameters['left_curverad'] = left_curverad if left_curverad is not None else 0
    parameters['right_curverad'] = right_curverad if right_curverad is not None else 0

    project = lane.project_back(undist_img)
    unwarp_project = unwarp(project, M, mtx, dist)
    show_text(project, parameters)
    if project is None:
        continue

    resize_project_img = resize_image(project, resize_shape, 'Project')
    resize_color_img = resize_image(wy_img, resize_shape, 'Y & W')
    resize_combined_img = resize_image(combined, resize_shape, 'Combined')
    resize_unwarped_img = resize_image(unwarp_img, resize_shape, 'unwrap img')
    resize_hls_img = resize_image(gray, resize_shape, 'hls')
    resize_rgb_img = resize_image(undist_img_enhance, resize_shape, 'undist_img_enhance')
    resize_color_unwarped = resize_image(unwarp_project, resize_shape, 'project_unwarped')
    resize_undist_img = resize_image(undist_img, resize_shape, 'undist')
    resize_fit_lane_img = resize_image(fit_lane_img, resize_shape, 'fit lane')
    resize_edge_img = resize_image(edge, resize_shape, 'edge')

    # print(resize_color_img.shape, resize_unwarped_img.shape, resize_project_img.shape, resize_combined_img.shape)
    final_img = np.vstack((np.hstack((resize_color_img, resize_color_unwarped, resize_rgb_img)), \
                           np.hstack((resize_combined_img, resize_unwarped_img, resize_undist_img)),
                           np.hstack((resize_project_img, resize_edge_img, resize_fit_lane_img))))
    cv2.imshow(w_name, final_img)

cap.release()
cv2.destroyAllWindows()

