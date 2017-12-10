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
cap = cv2.VideoCapture('challenge_video.mp4')
# cap = cv2.VideoCapture('harder_challenge_video.mp4')
# cap = cv2.VideoCapture('saveVideo.avi')

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

    orig = frame
    # gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # equ_img = clahe.apply(gray)
    # orig = cv2.cvtColor(equ_img, cv2.COLOR_GRAY2BGR)
    # print(orig.shape)


    undist_img, wy_img, gray, color, combined, unwarp_img, edge, undist_img_enhance= pipline(orig, parameters)
    scale = 0.4

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
                           np.hstack((resize_fit_lane_img, resize_edge_img, resize_project_img))))
    cv2.imshow(w_name, final_img)


    # cv2.imshow('unwarp', resize_img)
    # resize_img = cv2.resize(orig, resize_shape, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow(w_name, resize_img)

    # cv2.imshow('project', resize_img)
    # # c = np.dstack((np.zeros_like(hls_img), hls_img, np.zeros_like(hls_img)))*255
    # # resize_img = cv2.resize(c, resize_shape, interpolation=cv2.INTER_CUBIC)
    # # cv2.imshow('S', resize_img)
    # #
    # # c = np.dstack((np.zeros_like(rgb_img), np.zeros_like(rgb_img), rgb_img))*255
    # # resize_img = cv2.resize(c, resize_shape, interpolation=cv2.INTER_CUBIC)
    # # cv2.imshow('R', resize_img)
    #
    # c = np.dstack((s, s, s))*255
    # resize_img = cv2.resize(c, resize_shape, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('C', resize_img)
    #
    # c = np.dstack((combined, combined, combined))*255
    # resize_img = cv2.resize(c, resize_shape, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('Sobel', resize_img)

cap.release()
cv2.destroyAllWindows()

# import numpy as np
# import cv2
#
#
# def resize_image(img, shape, title = ''):
#     if len(img.shape) == 3:
#         stack = img
#     else:
#         stack = np.dstack((img, img, img))*255
#     show_line(stack, title)
#     resize_img = cv2.resize(stack, shape, interpolation=cv2.INTER_CUBIC)
#     return resize_img
#
# # cap = cv2.VideoCapture('project_video.mp4')
# # # cap = cv2.VideoCapture('challenge_video.mp4')
# cap = cv2.VideoCapture('harder_challenge_video.mp4')
#
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg = cv2.createBackgroundSubtractorMOG2()
#
# while(1):
#     ret, frame = cap.read()
#
#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#     mask_rbg = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
#     scale = 0.45
#
#     draw = frame & mask_rbg
#     resize_shape = (int(frame.shape[1]*scale), int(frame.shape[0]*scale))
#
#     s = resize_image(frame, resize_shape)
#     d = resize_image(fgmask, resize_shape)
#     e = resize_image(draw, resize_shape)
#     stack = np.vstack((s, d, e))
#     cv2.imshow('frame',stack)
#     k = cv2.waitKey(10) & 0xFF
#     if k == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg = cv2.createBackgroundSubtractorMOG2()
#
# while(1):
#     ret, frame = cap.read()
#
#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#
#     cv2.imshow('frame',fgmask)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()