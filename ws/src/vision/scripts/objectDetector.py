#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import torch
import numpy as np
import os
import pathlib
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
path_yolo = str(pathlib.Path(__file__).parent) + "/../models/yolov5m_Objects365.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_yolo, force_reload=False)
# model = torch.hub.load()

def detect(image):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)

    for *xyxy, conf, cls,names in results.pandas().xyxy[0].itertuples(index=False):
        # Normalized [0-1] ymin, xmin, ymax, xmax
        height = frame.shape[1]
        width = frame.shape[0]
        x1 = xyxy[1]
        x2 = xyxy[0]
        y1 = xyxy[3]
        y2 = xyxy[2]

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    cv2.imshow("annotated_image.jpg", frame)
    cv2.waitKey(0)


     
def yolo_run_inference_on_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame)

        output = {
            'detection_boxes': [],  # Normalized ymin, xmin, ymax, xmax
            'detection_classes': [], # ClassID 
            'detection_names': [], # Class Name
            'detection_scores': [] # Confidence
        }
        
        for *xyxy, conf, cls,names in results.pandas().xyxy[0].itertuples(index=False):
            # Normalized [0-1] ymin, xmin, ymax, xmax
            height = frame.shape[1]
            width = frame.shape[0]

            if conf < 0.5:
                continue
            output['detection_boxes'].append([xyxy[1]/width, xyxy[0]/height, xyxy[3]/width, xyxy[2]/height])
            output['detection_classes'].append(cls)
            output['detection_names'].append(names)
            output['detection_scores'].append(conf)

        output['detection_boxes'] = np.array(output['detection_boxes'])
        output['detection_classes'] = np.array(output['detection_classes'])
        output['detection_names'] = np.array(output['detection_names'])
        output['detection_scores'] = np.array(output['detection_scores'])
        return output

folder = "./images"
for filename in (os.listdir(folder)):

    path = os.path.join(folder, filename)
    img = cv2.imread(path)
    detect(img)
    print(filename)