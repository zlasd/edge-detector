######## Picamera Object Detection Using Tensorflow Classifier #########
#
# Author: Evan Juras
# Date: 4/15/18
# Description: 
# This program uses a TensorFlow classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. It also can be used with a webcam by adding "--usbcam"
# when executing this script from the terminal.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

# Set up camera constants
IM_WIDTH = 300
IM_HEIGHT = 300

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'


MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_GRAPH = os.path.join(CWD_PATH, MODEL_NAME, 'graph.pbtxt')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90


# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize Picamera and grab reference to the raw capture
camera = PiCamera()
camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
rawCapture.truncate(0)


cvNet = cv2.dnn.readNetFromTensorflow(PATH_TO_CKPT, PATH_TO_GRAPH)

for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

    t1 = cv2.getTickCount()
    
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = np.copy(frame1.array)
    frame.setflags(write=1)
    # frame_expanded = np.expand_dims(frame, axis=0)

    cvNet.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=False, crop=False))
    cvOut = cvNet.forward()

    # Draw the results of the detection (aka 'visulaize the results')
    for detection in cvOut[0,0,:,:]:
        # print(detection[1])
        # print(detection[2])
        classID = int(detection[1])
        score = float(detection[2])
        if classID == 1 and score > 0.3:
            left = detection[3] * IM_WIDTH
            top = detection[4] * IM_HEIGHT
            right = detection[5] * IM_WIDTH
            bottom = detection[6] * IM_HEIGHT
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (125, 255, 51), thickness=4)

    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    rawCapture.truncate(0)

camera.close()

cv2.destroyAllWindows()