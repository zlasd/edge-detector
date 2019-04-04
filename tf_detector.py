
# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import paho.mqtt.client as mqtt
import tensorflow as tf
import sys

import my_utils
from mqtt_client import client, sendMQTT


IM_WIDTH = 300
IM_HEIGHT = 300


MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'mscoco_label_map.pbtxt')
NUM_CLASSES = 90


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
    sess = tf.Session(graph=detection_graph)

    
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

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

buf_size, buf_index = 5, 0
ratios = np.zeros(buf_size)
whr = np.zeros(buf_size)

im_size = 5
imlist = []


fighting = False
record_deplay = 0

for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

    t1 = cv2.getTickCount()

    frame = np.copy(frame1.array)
    # frame = frame1.array
    frame.setflags(write=1)
    frame_expanded = np.expand_dims(frame, axis=0)

    # running the model
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # vis_util.visualize_boxes_and_labels_on_image_array(
        # frame,
        # np.squeeze(boxes),
        # np.squeeze(classes).astype(np.int32),
        # np.squeeze(scores),
        # category_index,
        # use_normalized_coordinates=True,
        # line_thickness=8,
        # min_score_thresh=0.40)

    num_detect = int(num[0])
    bboxs = np.zeros([num_detect, 4])
    confidence = 0.0
    count = 0
    for i in range(num_detect):
        classId = int(classes[0][i])
        score = float(scores[0][i])
        bbox = [float(v) for v in boxes[0][i]]
        if classId == 1 and score > 0.5:
            if (bbox[2]-bbox[0])*IM_HEIGHT/IM_WIDTH/(bbox[3]-bbox[1])<2:
                whr[buf_index] = 1
            bboxs[i] = bbox
            count += 1
            left = bbox[1] * IM_WIDTH
            top = bbox[0] * IM_HEIGHT
            right = bbox[3] * IM_WIDTH
            bottom = bbox[2] * IM_HEIGHT
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (125, 255, 51), thickness=4)
            
    if count > 1:
        ratios[buf_index] = my_utils.IoU(bboxs[0],bboxs[1])
        count = np.sum(ratios > 0.2)
        count_whrover = np.sum(whr == 1)
        if count >= 2 and count_whrover >= 3:
            my_utils.visualize_fighting(frame, bboxs)
            confidence = count * 0.11 + count_whrover * 0.12
            fighting = True
    else:
        ratios[buf_index] = 0
        whr[buf_index] = 0
    
    
    # cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

    print(frame_rate_calc)
    cv2.imshow('Edge Detector', frame)

    imlist.append(frame)
    if fighting:
        record_deplay += 1
        if record_deplay == 5:
            sendMQTT(imlist, confidence)
            fighting = False
            record_deplay = 0
            imlist = []
    elif len(imlist) > im_size:
        imlist.pop(0)
    
    buf_index = (buf_index + 1) % buf_size
    
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    if cv2.waitKey(1) == ord('q'):
        break
    rawCapture.truncate(0)

camera.close()
cv2.destroyAllWindows()
