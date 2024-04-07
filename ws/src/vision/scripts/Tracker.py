#!/usr/bin/env python3
import rospy
import time

import cv2
import numpy as np
import os

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO


CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"

class PersonTracker():

    def __init__(self):
        rospy.init_node('person_tracker')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        self.image = None
        self.model = YOLO('yolov8n.pt')

    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def run(self):
        while not rospy.is_shutdown():

            if self.image is not None:
                frame = self.image

                results = self.model.track(frame, persist=True, tracker='bytetrack.yaml', classes=0)
                frame_result = results[0].plot()

                cv2.imshow("Image", frame_result)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
            else:
                print("No image")

        cv2.destroyAllWindows()
            

node = PersonTracker()
node.run()
            