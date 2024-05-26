#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import torch
import numpy as np
import os
import pathlib
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
# path_yolo = str(pathlib.Path(__file__).parent) + "/../models/yolov5m_Objects365.pt"

path_yolo = str(pathlib.Path(__file__).parent) + "/../models/yolo11classes.pt"
yolo_v8 = str(pathlib.Path(__file__).parent) + "/../models/yolo"
# for filename in (os.listdir(f)):
#     print(filename)

# print("end")
useV8 = False
if useV8:
    model = YOLO('yolov8n.pt')
else:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_yolo, force_reload=False)
print("Loaded")
# model = torch.hub.load()

def detect(frame):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    output = {
        'detection_boxes': [],  # Normalized ymin, xmin, ymax, xmax
        'detection_classes': [], # ClassID 
        'detection_names': [], # Class Name
        'detection_scores': [] # Confidence
    }

    for *xyxy, conf, cls, names in results.pandas().xyxy[0].itertuples(index=False):
        # Normalized [0-1] ymin, xmin, ymax, xmax
        # if conf < 0.5:
        #     continue

        height = frame.shape[1]
        width = frame.shape[0]
        # output['detection_boxes'].append([xyxy[1]/width, xyxy[0]/height, xyxy[3]/width, xyxy[2]/height])
        output['detection_boxes'].append([xyxy[1], xyxy[0], xyxy[3], xyxy[2]])
        output['detection_classes'].append(cls)
        output['detection_names'].append(names)
        output['detection_scores'].append(conf)
        
    for box, name in zip(output['detection_boxes'], output['detection_names']):
        y1, x1, y2, x2 = box

        print(name)
        cv2.putText(frame, name, (int(x1 + 20), int(y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    cv2.imshow("annotated_image.jpg", frame)
    cv2.waitKey(0)

def yolo_v8_run(frame):
    # frame = self.image
    # self.output_img = frame
    
    results = model(frame, verbose=False)
    output = 0

    people = []
    chairs = []
    couches = []

    for out in results:
        for box in out.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            class_id = box.cls[0].item()
            label = model.names[class_id]
            bbox = (x1, y1, x2, y2)

            if class_id == 0:   
                people.append({"bbox": bbox, "label": label, "class": class_id})
            
            elif class_id == 56:
                chairs.append({"bbox": bbox, "label": label, "class": class_id})

            elif class_id == 57:
                couches.append({"bbox": bbox, "label": label, "class": class_id})

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # print(label)
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

folder = str(pathlib.Path(__file__).parent) + "/Utils/images"
for filename in (os.listdir(folder)):

    path = os.path.join(folder, filename)
    img = cv2.imread(path)
    print(filename)
    if useV8:

        yolo_v8_run(img)

    else:
        detect(img)