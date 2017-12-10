import os
import cv2
import glob

fps = 10

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('saveVideo.avi',fourcc,fps,(1280, 720))
images = glob.glob('./Capture/*.jpg')
for img in images:
    frame = cv2.imread(img)
    videoWriter.write(frame)
videoWriter.release()