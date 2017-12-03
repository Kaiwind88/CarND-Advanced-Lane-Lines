import numpy as np
import cv2
from math import *
import matplotlib.pyplot as plt
from enum import Enum
import pickle
import os.path
import glob


# key definition
# w->unwrap
# s->undistort
# c->color
# x->xsobel
# y->ysobel
# m->xysobel
# d->dirsobel
# left & right arrow -> thresh min
# up & down arrow -> thresh max
parameters = {
    'hlsthresh': [90, 255],
    'rgbthresh': [180, 255],
    'xthresh': [20, 100],
    'ythresh': [20, 100],
    'mthresh': [30, 100],
    'dthresh': [0.7, 1.2],
    'xksize': 5,
    'yksize': 5,
    'mksize': 9,
    'dksize': 15,
    'h': 'S',
    'b': 'R',
    'x': True,
    'y': True,
    'm': True,
    'd': True,
    'u': True,
    'w': True,
    'p': True,
    'mtx': None,
    'dist': None,
    'M': None,
    'MInv': None,
    'left_curverad': 0,
    'right_curverad': 0,
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
        M, MInv = transform(img, nx, ny, mtx, dist)
        parameters['mtx'] = mtx
        parameters['dist'] = dist
        parameters['M'] = M
        parameters['MInv'] = MInv

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

def unwarp(img, M, mtx, dist):
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
    ax1.set_title(orig_txt, fontsize=40)
    if dest.ndim == 3 and dest.shape[2] == 3:
        ax2.imshow(cv2.cvtColor(dest, cv2.COLOR_BGR2RGB))
    else:
        ax2.imshow(dest, cmap='gray')
    ax2.set_title(dest_txt, fontsize=40)
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


def hls_select(img, thresh=(0, 255), channel='S'):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if channel == 'S':
        ch_img = hls[:,:,2]
    elif channel == 'L':
        ch_img = hls[:,:,1]
    elif channel == 'H':
        ch_img = hls[:,:,0]
    else:
        ch_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Apply a threshold to the S hlschannel
    binary_output = np.zeros_like(ch_img)
    binary_output[(ch_img > thresh[0]) & (ch_img <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


def rgb_select(img, thresh=(0, 255), channel='R'):
    # 1) Convert to HLS color space
    if channel == 'R':
        ch_img = img[:,:,2]
    elif channel == 'G':
        ch_img = img[:,:,1]
    elif channel == 'B':
        ch_img = img[:,:,0]
    else:
        ch_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Apply a threshold to the S hlschannel
    binary_output = np.zeros_like(ch_img)
    binary_output[(ch_img > thresh[0]) & (ch_img <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output



def pipline(img, parameters):

    # Step 1: undistort
    sw = parameters['u']
    mtx = parameters['mtx']
    dist = parameters['dist']
    undist_img = cal_undistort(img, mtx, dist) if sw else img

    # Step 2: convert color space
    thresh = parameters['hlsthresh']
    channel = parameters['h']
    hls_img = hls_select(undist_img, thresh=thresh, channel=channel)

    thresh = parameters['rgbthresh']
    channel = parameters['b']
    rgb_img = rgb_select(undist_img, thresh=thresh, channel=channel)

    single_channel_img = np.zeros_like(rgb_img)
    single_channel_img[(rgb_img == 1) | (hls_img == 1)] = 1
    # Step 2: xsobel
    thresh = parameters['xthresh']
    ksize = parameters['xksize']
    sw = parameters['x']
    xsobel = abs_sobel_thresh(single_channel_img, orient='x', sobel_kernel=ksize, thresh=thresh)\
        if sw else np.zeros_like(single_channel_img)

    # Step 3: ysobel
    thresh = parameters['ythresh']
    ksize = parameters['yksize']
    sw = parameters['y']
    ysobel = abs_sobel_thresh(single_channel_img, orient='y', sobel_kernel=ksize, thresh=thresh)\
        if sw else np.zeros_like(single_channel_img)

    # Step 4: msobel
    thresh = parameters['mthresh']
    ksize = parameters['mksize']
    sw = parameters['m']
    msobel = mag_thresh(single_channel_img, sobel_kernel=ksize, mag_thresh=thresh)\
        if sw else np.zeros_like(single_channel_img)

    # Step 5: dsobel
    thresh = parameters['dthresh']
    ksize = parameters['dksize']
    sw = parameters['d']
    dsobel = dir_thresh(single_channel_img, sobel_kernel=ksize, thresh=thresh)\
        if sw else np.zeros_like(single_channel_img)

    combined = np.zeros_like(single_channel_img)
    combined[((xsobel == 1) & (ysobel == 1)) | ((msobel == 1) & (dsobel == 1))] = 1

    return combined, undist_img, single_channel_img, hls_img, rgb_img, xsobel, ysobel, msobel, dsobel


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
    elif key == ord('b'):
        parameters['b'] = parameters_range['rgbchannel'][key_handler.rgbchannel]
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
    elif key in arrow_key_code.values():
        if key_handler.arrow_key_state.value != key_state.ignore.value:
            parameters_key = key_handler.arrow_key_state.name
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

            parameters[parameters_key][0] = thresh_min
            parameters[parameters_key][1] = thresh_max
    elif key != 0xFF:
        key_handler.arrow_key_state = key_state.ignore

    return False

def show_text(img, parameters):
    lines = []
    undisort_switch = 'ON' if parameters['u'] else 'OFF'
    unwarp_switch = 'ON' if parameters['w'] else 'OFF'
    text = "undisort:{} unwarp:{}".format(undisort_switch, unwarp_switch)
    lines.append(text)
    text = "HLS channel:{} min:{} max:{}".format(parameters['h'], parameters['hlsthresh'][0], parameters['hlsthresh'][1])
    lines.append(text)
    text = "RGB channel:{} min:{} max:{}".format(parameters['b'], parameters['rgbthresh'][0], parameters['rgbthresh'][1])
    lines.append(text)
    for prefix in ['x', 'y', 'm', 'd']:
        thresh_name = prefix+'thresh'
        switch = 'ON' if parameters[prefix] else 'OFF'
        text = "{}thresh:{} min:{} max:{}".format(prefix, switch, parameters[thresh_name][0], parameters[thresh_name][1])
        lines.append(text)
    text = "Left Curvature:{:>.4f}m".format(parameters['left_curverad'])
    lines.append(text)
    text = 'Right Curvature:{:>.4f}m'.format(parameters['right_curverad'])
    lines.append(text)

    for i, line in enumerate(lines):
        x = 1
        y = 20 + 25*i
        cv2.putText(img, text=line, org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, \
                fontScale=1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)


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
    src = np.float32([[579, 460], [704, 460], [1110, 719.5], [200, 719.5]])
    dst = np.float32([[320, 0.5], [960, 0.5], [960, 719.5], [320, 719.5]])
    M = get_perspective_transform(img, nx, ny, mtx, dist, src, dst)
    MInv = get_perspective_transform(img, nx, ny, mtx, dist, dst, src)
    return M, MInv
