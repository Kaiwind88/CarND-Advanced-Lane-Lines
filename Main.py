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
parameters['p'] = False
parameters['b'] = False

f1 = 'project_video.mp4'
f2 = 'challenge_video.mp4'
f3 = 'harder_challenge_video.mp4'

# Change Input File Here

input_file = f3
if input_file == f3:
    parameters['M'] = M_mid
    parameters['MInv'] = MInv_mid
    parameters['color_sw'] = False
    parameters['x'] = True
    parameters['y'] = True
    parameters['m'] = False
    parameters['d'] = False
    parameters['margin'] = 65
else:
    parameters['M'] = M_max
    parameters['MInv'] = MInv_max
    parameters['color_sw'] = True
    parameters['x'] = False
    parameters['y'] = False
    parameters['m'] = False
    parameters['d'] = False
    parameters['margin'] = 40

lane = Lane(parameters)
lane_verify = Lane(parameters)

w_name = input_file
cv2.namedWindow(w_name, cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(input_file)

def progress_bar_cb(x):
    cap.set(cv2.CAP_PROP_POS_FRAMES, x)
cv2.createTrackbar('Frame',w_name,0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),progress_bar_cb)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
prefix = input_file.split('.')[0]
out_debug = cv2.VideoWriter(prefix+'_debug.avi',fourcc, 20.0, (1152, 864))
out_project = cv2.VideoWriter(prefix+'_project.avi',fourcc, 20.0, (640, 360))
delay = 1
while(cap.isOpened()):
    if key_handler(delay, parameters):
        break
    if parameters['s']:
        delay = 0
    else:
        delay = 1

    if not parameters['p']:
        ret, frame = cap.read()
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos('Frame',w_name, frame_idx)
        set_frame_idx(frame_idx)
        parameters['frame'] = frame_idx
        if not ret:
            continue

    print("--------current frame:", frame_idx)

    final_img, project = find_lane(frame, lane, lane_verify)
    print(final_img.shape)
    cv2.imshow(w_name, final_img)
    out_debug.write(final_img)
    out_project.write(project)

cap.release()
cv2.destroyAllWindows()

