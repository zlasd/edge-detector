import os
import cv2 as cv
import numpy as np
import random

import tensorflow as tf
import argparse

import tf_utils
import my_utils

"""
https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py
"""

MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
# MODEL_NAME = 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03'
# MODEL_NAME = 'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, 'models', MODEL_NAME, 'frozen_inference_graph.pb')
IMG_FOLDER = os.path.join('data', 'test_img')
IMG_NAME = random.choice(os.listdir(IMG_FOLDER))
#IMG_PATH = os.path.join('env/img', 'zhanghua_0006.jpg')
LABELS_PATH = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

label_map = tf_utils.load_labelmap(LABELS_PATH)
categories = tf_utils.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = tf_utils.create_category_index(categories)

# print(label_map)
# print(categories)
# print(category_index)

def visualize_fighting(img, bboxs):
    """ visualize fighting bounding box
    """
    rows, cols = img.shape[:2]
    top = min(bboxs[0][0], bboxs[1][0]) * rows
    left = min(bboxs[0][1], bboxs[1][1]) * cols
    bottom = max(bboxs[0][2], bboxs[1][2]) * rows
    right = max(bboxs[0][3], bboxs[1][3]) * cols
    cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0xff, 0xcc, 0x66), thickness=10)
    




with tf.gfile.FastGFile(PATH_TO_CKPT, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())



with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    FileNames = os.listdir(os.path.join(CWD_PATH, IMG_FOLDER))
    img_num=len(FileNames)
    ratios=np.zeros(img_num)
    whr=np.zeros(img_num)
    
    for j in range(img_num):
        IMG_PATH = os.path.join(IMG_FOLDER, FileNames[j])
        # Read and preprocess an image.
        img = cv.imread(IMG_PATH)
        rows = img.shape[0]
        cols = img.shape[1]
        # inp = cv.resize(img, (300, 300))
        inp = img[::3, ::3, :]
        # inp = img[:,:,:]
        # inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        tf_utils.visualize_boxes_and_labels_on_image_array(
            inp,
            np.squeeze(out[2]),
            np.squeeze(out[3]).astype(np.int32),
            np.squeeze(out[1]),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)
        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        bboxs=np.zeros([num_detections,4])
        count=0
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if classId==1:
                if (bbox[2]-bbox[0])*rows/cols/(bbox[3]-bbox[1])<2:
                    whr[j]=1
                bboxs[count]=bbox
                count=count+1
            # if score > 0.3:
                # left = bbox[1] * cols
                # top = bbox[0] * rows
                # right = bbox[3] * cols
                # bottom = bbox[2] * rows
                # cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0xff, 0xcc, 0x66), thickness=10)
        if(len(bboxs)>1):
            ratios[j]=my_utils.IoU(bboxs[0],bboxs[1])
            count = np.sum(ratios[j - 5:j] > 0.1)
            count_whrover = np.sum(whr[j - 5:j] == 1)
            if count >= 2 and count_whrover >= 3:
                visualize_fighting(img, bboxs)
        
        cv.imshow('TensorFlow MobileNet-SSD', inp)
        cv.waitKey()
varance=my_utils.var(ratios)
