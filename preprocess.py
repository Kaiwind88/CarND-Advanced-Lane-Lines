import numpy as np
import cv2
from math import *
import matplotlib.pyplot as plt
from enum import Enum
import pickle
import os.path
import glob

# key definition
# w->unwarp
# s->undistort
# c->color
# x->xsobel
# y->ysobel
# m->xysobel
# d->dirsobel
# left & right arrow -> thresh min
# up & down arrow -> thresh max
parameters = {
    'hlsthresh': [[0, 40], [145, 255], [80, 255], [0, 255]],
    'rgbthresh': [[200, 255], [200, 255], [200, 255], [0, 255]],
    'graythresh': [0, 255],
    'xthresh': [26, 100],
    'ythresh': [26, 100],
    'mthresh': [30, 100],
    'dthresh': [0.7, 1.2],
    'xksize': 3,
    'yksize': 3,
    'mksize': 9,
    'dksize': 15,
    'h': 'S',
    'c': 'B',
    'b': False,
    'x': True,
    'y': True,
    'm': True,
    'd': True,
    'u': True,
    'w': True,
    'p': True,
    's': False,
    'r': True,
    'mtx': None,
    'dist': None,
    'M_max': None,
    'MInv_max': None,
    'M_mid': None,
    'MInv_mid': None,
    'M_min': None,
    'MInv_min': None,
    'M': None,
    'MInv': None,
    'left_curverad': 0,
    'right_curverad': 0,
    'config-option': 0,
    'brightness': 0,
    'use_color': False,
    'color_sw': True,
    'margin': 40,
    'offset': 0
}


parameters_range = {
    'hlsthresh': (0, 255, 2),
    'rgbthresh': (0, 255, 2),
    'xthresh': (0, 255, 2),
    'ythresh': (0, 255, 2),
    'mthresh': (0, 255, 2),
    'dthresh': (0, np.pi/2, 0.2),
    'hlschannel': ('H', 'L', 'S', 'Gray'),
    'rgbchannel': ('R', 'G', 'B', 'Gray'),
}

pickle_file = 'data.pickle'
def load_data():
    global parameters
    nx = 9
    ny = 6
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            param = pickle.load(f)
            parameters.update(param)
    else:
        images = glob.glob('./camera_cal/calibration*.jpg')
        mtx, dist = calibration(images, nx, ny)
        img = cv2.imread('./test_images/straight_lines1.jpg')
        M_max, MInv_max, M_mid, MInv_mid, M_min, MInv_min= transform(img, nx, ny, mtx, dist)
        parameters['mtx'] = mtx
        parameters['dist'] = dist
        parameters['M_max'] = M_max
        parameters['MInv_max'] = MInv_max
        parameters['M_mid'] = M_mid
        parameters['MInv_mid'] = MInv_mid
        parameters['M_min'] = M_min
        parameters['MInv_min'] = MInv_min

        with open(pickle_file, 'wb') as f:
            pickle.dump(parameters, f, pickle.HIGHEST_PROTOCOL)

def save_data():
    global parameters
    with open(pickle_file, 'wb') as f:
        pickle.dump(parameters, f, pickle.HIGHEST_PROTOCOL)
        print('Save Parameters')


def calibration(images, nx, ny):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist


def cal_undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def show_comparison_images(orig, dest):
    comp = np.hstack((orig, dest))
    comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 8))
    plt.imshow(comp)
    plt.show()

def breakpoint(b):
    if b and parameters['b']:
        parameters['s'] = True

def region_of_interest(img):
    # return img
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    # Define a four sided polygon to mask edges image
    height, width = img.shape[:2]
    v_bottom_left = (int(width*0.16), int(height*0.95))
    v_bottom_right = (int(width*0.94), int(height*0.95))
    v_top_left = (int(width*0.47), int(height*0.59))
    v_top_right = (int(width*0.58), int(height*0.59))
    src = np.float32([[579, 460], [704, 460], [1110, 719.5], [200, 719.5]])
    dst = np.float32([[320, 0.5], [960, 0.5], [960, 719.5], [320, 719.5]])

    vertices = np.array([[v_bottom_left, v_top_left, v_top_right, v_bottom_right]], dtype=np.int32)
    # cv2.polylines(img, vertices, True, (0,0,255))

    mask = np.zeros_like(img, dtype=np.uint8)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def filter_yellow_white_color(img):
    img_cov = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)

    if parameters['use_color']:
        y_upper = np.array([80, 255, 255])
        y_lower = np.array([0, 10, 20])
        w_upper = np.array([255, 255, 255])
        w_lower = np.array([0, 195, 0])
    else:
        # No 2
        y_upper = np.array([80, 255, 255])
        y_lower = np.array([0, 10, 20])
        w_upper = np.array([255, 255, 255])
        w_lower = np.array([0, 130, 0])

    y_img_mask = cv2.inRange(img_cov, y_lower, y_upper)
    w_img_mask = cv2.inRange(img_cov, w_lower, w_upper)
    wy_img_mask = cv2.bitwise_or(y_img_mask, w_img_mask)
    # img_new = np.ones_like(img) * 255
    wy_img = cv2.bitwise_and(img, img, mask=wy_img_mask)
    return wy_img

def test_brightness(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # equ = cv2.equalizeHist(hls[:,:,2])
    roi = region_of_interest(hls[:,:,2])
    nonzero = np.zeros_like(roi)
    nonzero[roi > 0] = 1
    print(roi.shape)
    cnt = np.sum(nonzero)
    brightness = np.sum(roi) / cnt

    return brightness


def get_perspective_transform(img, nx, ny, mtx, dist, src = None, dst = None):
    undist = cal_undistort(img, mtx, dist)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    M = None
    if src is None and dst is None:
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            offset = 105
            img_size = (gray.shape[1], gray.shape[0])
            src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
            dst = np.float32([[offset, offset], [img_size[0]-offset, offset],\
                              [img_size[0]-offset, img_size[1]-offset],\
                              [offset, img_size[1]-offset]])
            M = cv2.getPerspectiveTransform(src, dst)
            print(src)
            print(dst)
    else:
        M = cv2.getPerspectiveTransform(src, dst)
    return M

def warp_perspective(img, M, mtx, dist):
    undist = cal_undistort(img, mtx, dist)
    img_size = (int(img.shape[1]), int(img.shape[0]))
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped

def show_images(orig, dest, orig_txt='Original Image', dest_txt = 'Destination Image'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
    f.tight_layout()
    if orig.ndim == 3 and orig.shape[2] == 3:
        ax1.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(orig, cmap='gray')
    ax1.set_title(orig_txt, fontsize=20)
    if dest.ndim == 3 and dest.shape[2] == 3:
        ax2.imshow(cv2.cvtColor(dest, cv2.COLOR_BGR2RGB))
    else:
        ax2.imshow(dest, cmap='gray')
    ax2.set_title(dest_txt, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3,  thresh=(0, 255)):
    # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = img
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def color_select(img, channel='R', parameters=parameters):
    # 1) Convert to HLS color space
    HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if channel == 'R':
        ch_img = img[:,:,2]
        thresh = parameters['rgbthresh'][0]
    elif channel == 'G':
        ch_img = img[:,:,1]
        thresh = parameters['rgbthresh'][1]
    elif channel == 'B':
        ch_img = img[:,:,0]
        thresh = parameters['rgbthresh'][2]
    elif channel == 'H':
        ch_img = HLS[:,:,0]
        thresh = parameters['hlsthresh'][0]
    elif channel == 'L':
        ch_img = HLS[:,:,1]
        thresh = parameters['hlsthresh'][1]
    elif channel == 'S':
        ch_img = HLS[:,:,2]
        thresh = parameters['hlsthresh'][2]
    else:
        ch_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = parameters['graythresh']

    binary_output = np.zeros_like(ch_img)
    binary_output[(ch_img > thresh[0]) & (ch_img <= thresh[1])] = 1
    return binary_output


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

from collections import deque
bright_deque = deque(maxlen=5)
def adjust_parameter(parameters=parameters):
    if parameters['p']:
        return
    brightness = np.mean(bright_deque)
    if brightness >= 20 and brightness <= 50:
        ch = 2 #hls -> s
        s_factor = 3.0
        s_min = brightness * s_factor
        parameters['h'] = parameters_range['hlschannel'][ch]
        parameters['hlsthresh'][ch][0] = int(s_min if s_min < 130 else 130)
    elif brightness < 20:
        ch = 0 #hls -> h
        h_factor = 700
        h_max = h_factor / brightness
        parameters['h'] = parameters_range['hlschannel'][ch]
        parameters['hlsthresh'][ch][1] = int(h_max if h_max > 10 else 10)
    elif brightness > 50:
        ch = 1 #hls->l
        l_factor = 60
        l_min = brightness + l_factor
        parameters['h'] = parameters_range['hlschannel'][ch]
        parameters['hlsthresh'][ch][0] = int(l_min if l_min < 170 else 170)

    b_factor = 180
    b_min = int(brightness + b_factor)
    parameters['rgbthresh'][2][0] = b_min if b_min < 230 else 230

def otsu(img):
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plot all the images and their histograms
    return th

def enhance_img(img):
    dst = cv2.GaussianBlur(img, (0,0), 3)
    out = cv2.addWeighted(img, 1.5, dst, -0.5, 0)
    return out

def use_color(use):
    parameters['use_color'] = use

def preprocessing(img, M, parameters=parameters):
    kernel = np.ones((1, 1), np.uint8)
    # Step 1: undistort
    sw = parameters['u']
    mtx = parameters['mtx']
    dist = parameters['dist']
    undist_img = cal_undistort(img, mtx, dist) if sw else img

    # Step 2: Enhence Image
    undist_img_enhance = enhance_img(np.copy(undist_img))

    # Step 3: Get the brightness
    parameters['brightness'] = test_brightness(undist_img_enhance)
    bright_deque.append(parameters['brightness'])
    # parameters['brightness'] = np.mean(bright_deque)
    adjust_parameter()

    # Step 4: First filter yellow and white color
    input = undist_img_enhance
    wy_img = filter_yellow_white_color(input)

    # Step 5: White 2nd filter based on first filter
    channel = parameters['c']
    rgb_img = color_select(wy_img, channel=channel)

    # Step 6: Yellow 2nd filter based on first filter
    channel = parameters['h']
    hls_img = color_select(wy_img, channel=channel)

    # Step 7: Combined Color
    color = np.zeros_like(rgb_img)
    color[(rgb_img == 1) | (hls_img == 1)] = 1
    color = cv2.dilate(color, kernel, iterations=1)

    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    zero_img = np.zeros_like(gray)
    soble_input = gray
    soble_input = gaussian_blur(soble_input, 15)
    # Step 7: xsobel
    thresh = parameters['xthresh']
    ksize = parameters['xksize']
    sw = parameters['x']
    xsobel = abs_sobel_thresh(soble_input, orient='x', sobel_kernel=ksize, thresh=thresh)\
        if sw else zero_img
    # xsobel = cv2.dilate(xsobel, kernel, iterations = 1)

    # Step 8: ysobel
    thresh = parameters['ythresh']
    ksize = parameters['yksize']
    sw = parameters['y']
    ysobel = abs_sobel_thresh(soble_input, orient='y', sobel_kernel=ksize, thresh=thresh)\
        if sw else zero_img
    # ysobel = cv2.dilate(ysobel, kernel, iterations = 1)

    # Step 9: msobel
    thresh = parameters['mthresh']
    ksize = parameters['mksize']
    sw = parameters['m']
    msobel = mag_thresh(soble_input, sobel_kernel=ksize, mag_thresh=thresh)\
        if sw else zero_img

    # Step 10: dsobel
    thresh = parameters['dthresh']
    ksize = parameters['dksize']
    sw = parameters['d']
    dsobel = dir_thresh(soble_input, sobel_kernel=ksize, thresh=thresh)\
        if sw else zero_img

    edge = np.zeros_like(gray)
    edge[(((xsobel == 1) & (ysobel == 1)) | ((msobel == 1) & (dsobel == 1)))] = 1
    # edge = unwarp(edge, M, mtx, dist)

    if not parameters['use_color']:
        color_img = zero_img
    else:
        color_img = color

    # Step 11: Combined All
    combined = np.zeros_like(gray)
    combined[(((xsobel == 1) & (ysobel == 1)) | ((msobel == 1) & (dsobel == 1))) | (color_img == 1)] = 1
    # combined = cv2.dilate(combined, kernel, iterations=1)

    mtx = parameters['mtx']
    dist = parameters['dist']
    combine_warped = warp_perspective(combined, M, mtx, dist)

    return combine_warped, undist_img, wy_img, gray, color, combined, edge, undist_img_enhance, rgb_img, hls_img

# def enum(*sequential):
#     enums = dict(zip(sequential, range(len(sequential))))
#     return type('Enum', (), enums)

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(arrow_key_state=None, hlschannel=0, rgbchannel=0)
def key_handler(delay, parameters):
    arrow_key_code = {'left': ord('9'), 'right': ord('0'), 'up': ord('='), 'down': ord('-')}

    class key_state(Enum):
        ignore = 0
        xthresh = 1
        ythresh = 2
        mthresh = 3
        dthresh = 4
        hlsthresh = 5
        rgbthresh = 6

    if key_handler.arrow_key_state is None:
        key_handler.arrow_key_state = key_state.ignore

    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        return True
    elif key == ord('h'):
        parameters['h'] = parameters_range['hlschannel'][key_handler.hlschannel]
        key_handler.hlschannel += 1
        key_handler.hlschannel = key_handler.hlschannel % 4
        key_handler.arrow_key_state = key_state.hlsthresh
    elif key == ord('c'):
        parameters['c'] = parameters_range['rgbchannel'][key_handler.rgbchannel]
        key_handler.rgbchannel += 1
        key_handler.rgbchannel = key_handler.rgbchannel % 4
        key_handler.arrow_key_state = key_state.rgbthresh
    elif key == ord('x'):
        parameters['x'] = not parameters['x']
        key_handler.arrow_key_state = key_state.xthresh if parameters['x'] else key_state.ignore
    elif key == ord('y'):
        parameters['y'] = not parameters['y']
        key_handler.arrow_key_state = key_state.ythresh if parameters['y'] else key_state.ignore
    elif key == ord('m'):
        parameters['m'] = not parameters['m']
        key_handler.arrow_key_state = key_state.mthresh if parameters['m'] else key_state.ignore
    elif key == ord('d'):
        parameters['d'] = not parameters['d']
        key_handler.arrow_key_state = key_state.dthresh if parameters['d'] else key_state.ignore
    elif key == ord('u'):
        parameters['u'] = not parameters['u']
        key_handler.arrow_key_state = key_state.ignore
    elif key == ord('w'):
        parameters['w'] = not parameters['w']
        key_handler.arrow_key_state = key_state.ignore
    elif key == ord('p'):
        parameters['p'] = not parameters['p']
        key_handler.arrow_key_state = key_state.ignore
    elif key == ord('S'):
        save_data()
        key_handler.arrow_key_state = key_state.ignore
    elif key == ord('s'):
        parameters['s'] = True
        parameters['r'] = False
        key_handler.arrow_key_state = key_state.ignore
    elif key == ord('r'):
        parameters['r'] = True
        parameters['s'] = False
        key_handler.arrow_key_state = key_state.ignore
    elif key == ord('b'):
        parameters['b'] = not parameters['b']
        key_handler.arrow_key_state = key_state.ignore
    elif key == ord('C'):
        parameters['use_color'] = not parameters['use_color']
        key_handler.arrow_key_state = key_state.ignore
    elif key in arrow_key_code.values():
        if key_handler.arrow_key_state.value != key_state.ignore.value:
            parameters_key = key_handler.arrow_key_state.name
            if parameters_key == 'hlsthresh':
                ch = parameters_range['hlschannel'].index(parameters['h'])
                thresh_min = parameters[parameters_key][ch][0]
                thresh_max = parameters[parameters_key][ch][1]
            elif  parameters_key == 'rgbthresh':
                ch = parameters_range['rgbchannel'].index(parameters['c'])
                thresh_min = parameters[parameters_key][ch][0]
                thresh_max = parameters[parameters_key][ch][1]
            else:
                thresh_min = parameters[parameters_key][0]
                thresh_max = parameters[parameters_key][1]

            range_min = parameters_range[parameters_key][0]
            range_max = parameters_range[parameters_key][1]
            step = parameters_range[parameters_key][2]
            if key == arrow_key_code['left']:
                thresh_min -= step
            elif key == arrow_key_code['right']:
                thresh_min += step
            elif key == arrow_key_code['up']:
                thresh_max += step
            elif key == arrow_key_code['down']:
                thresh_max -= step

            if thresh_min > range_max:
                thresh_min = range_max
            elif thresh_min < range_min:
                thresh_min = range_min

            if thresh_max > range_max:
                thresh_max = range_max
            elif thresh_max < range_min:
                thresh_max = range_min

            if thresh_min > thresh_max:
                thresh_min = thresh_max

            if parameters_key == 'hlsthresh':
                ch = parameters_range['hlschannel'].index(parameters['h'])
                parameters[parameters_key][ch][0] = thresh_min
                parameters[parameters_key][ch][1] = thresh_max
            elif parameters_key == 'rgbthresh':
                parameters[parameters_key][ch][0] = thresh_min
                parameters[parameters_key][ch][1] = thresh_max
            else:
                ch = parameters_range['rgbchannel'].index(parameters['c'])
                parameters[parameters_key][0] = thresh_min
                parameters[parameters_key][1] = thresh_max

    elif key != 0xFF:
        key_handler.arrow_key_state = key_state.ignore

    return False

def show_text(img, parameters):
    lines = []
    # undisort_switch = 'ON' if parameters['u'] else 'OFF'
    # unwarp_switch = 'ON' if parameters['w'] else 'OFF'
    # text = "Undisort:{} unwarp:{}".format(undisort_switch, unwarp_switch)
    # lines.append(text)
    ch = parameters_range['hlschannel'].index(parameters['h'])
    text = "HLS channel:{} min:{} max:{}".format(parameters['h'], \
            parameters['hlsthresh'][ch][0], parameters['hlsthresh'][ch][1])
    lines.append(text)
    ch = parameters_range['rgbchannel'].index(parameters['c'])
    text = "RGB channel:{} min:{} max:{}".format(parameters['c'], \
            parameters['rgbthresh'][ch][0], parameters['rgbthresh'][ch][1])
    lines.append(text)
    text = "Color Filter:{}".format('ON' if parameters['use_color'] else 'OFF')
    lines.append(text)
    for prefix in ['x', 'y', 'm', 'd']:
        thresh_name = prefix+'thresh'
        switch = 'ON' if parameters[prefix] else 'OFF'
        text = "{}thresh:{} min:{} max:{}".format(prefix, switch, parameters[thresh_name][0], parameters[thresh_name][1])
        lines.append(text)
    text = "Left Curvature:{:>.2f}m".format(parameters['left_curverad'])
    lines.append(text)
    text = 'Right Curvature:{:>.2f}m'.format(parameters['right_curverad'])
    lines.append(text)
    text = 'Offset:{} | {:>.2f}cm'.format('Left' if parameters['offset'] > 0 else 'Right', abs(parameters['offset']*100))
    lines.append(text)
    text = "S Channel ROI Brightness:{:>.2f}".format(parameters['brightness'])
    lines.append(text)
    text = "Breakpoints:{}".format('ON' if parameters['b'] else 'OFF')
    lines.append(text)

    for i, line in enumerate(lines):
        x = 10
        y = 30 + 40*i
        cv2.putText(img, text=line, org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, \
                fontScale=2, color=(255, 255, 00), thickness=2, lineType=cv2.LINE_AA)

def show_line(img, line):
    x = 10
    y = 670
    cv2.putText(img, text=line, org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, \
                fontScale=3, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

def show_ax_image(ax, img, title=''):
    if img.ndim == 3 and img.shape[2] == 3:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=40)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

def show_processed_images(image_dict):
    n_img = len(image_dict)
    cols = 4
    rows = ceil(n_img / cols)
    f, axs = plt.subplots(rows, cols, figsize=(12, 9))
    for i, (k, v) in enumerate(image_dict.items()):
        col = i % cols
        row = int(i / cols)
        ax = axs[row, col]
        show_ax_image(ax, v[1], v[0])

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def transform(img, nx, ny, mtx, dist):
    src = np.float32([[410, 560], [850, 560], [1110, 720], [200, 720]]) # (lt,rt,rb,lb)
    dst = np.float32([[320, 0], [960, 0], [960, 700], [320, 700]])

    M_min = get_perspective_transform(img, nx, ny, mtx, dist, src, dst)
    MInv_min = get_perspective_transform(img, nx, ny, mtx, dist, dst, src)


    src = np.float32([[480, 500], [740, 500], [1110, 720], [200, 720]]) # (lt,rt,rb,lb)
    dst = np.float32([[320, 0], [960, 0], [960, 700], [320, 700]])

    M_mid = get_perspective_transform(img, nx, ny, mtx, dist, src, dst)
    MInv_mid = get_perspective_transform(img, nx, ny, mtx, dist, dst, src)

    # src = np.float32([[579, 460], [704, 460], [1110, 719.5], [200, 719.5]])
    # dst = np.float32([[320, 0.5], [960, 0.5], [960, 719.5], [320, 719.5]])
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [((img_size[0] / 6) - 10), img_size[1]]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] * 3 / 4), 0],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] / 4), img_size[1]]])
    print(src, dst)
    M_max = get_perspective_transform(img, nx, ny, mtx, dist, src, dst)
    MInv_max = get_perspective_transform(img, nx, ny, mtx, dist, dst, src)
    return M_max, MInv_max, M_mid, MInv_mid, M_min, MInv_min

def resize_image(img, shape, title = ''):
    if len(img.shape) == 3:
        stack = img
    else:
        stack = np.dstack((img, img, img))*255
    show_line(stack, title)
    resize_img = cv2.resize(stack, shape, interpolation=cv2.INTER_CUBIC)
    h = resize_img.shape[0]-1
    w = resize_img.shape[1]-1
    cv2.rectangle(resize_img, (0, 0), (w, h), (0, 0, 255), 1)
    return resize_img


current_frame_idx = 0
def set_frame_idx(idx):
    global current_frame_idx
    current_frame_idx = idx

def get_frame_idx(idx):
    return current_frame_idx