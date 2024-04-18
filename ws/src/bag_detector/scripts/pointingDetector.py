#!/usr/bin/env python3
import rospy
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
from math import acos, degrees
from enum import Enum

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int32, Bool
from bag_detector.srv import Pointing


POINTING_TOPIC = "/detectios/pointing"
CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"
RES_TOPIC = "/res"

class Direction(Enum):
    NOT_POINTING = 0
    RIGHT = 1
    LEFT = 2

class PointingDetector:
    def __init__(self):
        rospy.init_node('pointing_detector')
        self.bridge = CvBridge()
        self.pointing_service = rospy.Service(POINTING_TOPIC, Pointing, self.pointing)
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)

        self.results_pub = rospy.Publisher(RES_TOPIC, Int32, queue_size=1)

        def load_models():
            self.model = YOLO('yolov8n.pt')
            self.pose_model = mp.solutions.pose.Pose(min_detection_confidence=0.8)

        load_models()
        self.image = None
        self.start = False
        self.run()

    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def pointing(self, req):
        self.start = req.active
    
    def getAngle(self, point_close, point_mid, point_far):
        # Convert the points to numpy arrays
        p1 = np.array([point_close.x, point_close.y])
        p2 = np.array([point_mid.x, point_mid.y])
        p3 = np.array([point_far.x, point_far.y])

        # Euclidean distances
        l1 = np.linalg.norm(p2 - p3)    #lados de triangulo
        l2 = np.linalg.norm(p1 - p3)
        l3 = np.linalg.norm(p1 - p2)

        # Cálculo de ángulo pierna izquierda
        return abs(degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3))))

    def getDirection(self, image):
        results = self.poseModel.process(image)

        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            shoulder_right = landmarks[12]
            shoulder_left = landmarks[11]
            wrist_left = landmarks[15]
            wrist_right = landmarks[16]
            index_right = landmarks[20]
            hip_right = landmarks[24]
            elbow_right = landmarks[14]

            x_center = (shoulder_right.x + shoulder_left.x) / 2

            angle_body = self.getAngle(hip_right, shoulder_right, elbow_right)
            arm_angle = self.getAngle(shoulder_right, elbow_right, wrist_right)

            if angle_body < 30:
                if arm_angle > 150 and arm_angle < 200:
                    return Direction.NOT_POINTING

                else:
                    if index_right.x > elbow_right.x:
                        return Direction.RIGHT
                
            else:
                if index_right.x > shoulder_right.x:
                    return Direction.RIGHT

                else:
                    return Direction.LEFT




            # y_center = (shoulder_right.y + shoulder_left.y) / 2

    def run(self):

        while rospy.is_shutdown() == False :
            if self.start:

                if self.image is not None:
                    print("img")
                    frame = self.image
                    results = self.model(frame, verbose=False, classes=[0])

                    max_area = 0
                    max_bbox = None

                    for out in results:
                        for box in out.boxes:
                            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                            bbox = (x1, y1, x2, y2)
                            w = (x2 - x1) 
                            h = (y2 - y1)
                            area = w*h

                            if area > max_area:
                                max_area = area
                                max_bbox = bbox

                    # x1, y1, x2, y2 = max_bbox
                    # crop = frame[y1:y2, x1:x2]
                    # pointing_direction = self.getDirection(bbox)

                    # self.results_pub.publish(pointing_direction)


                else:
                    rospy.loginfo("No image provided")

            # else:
            #     rospy.loginfo("No image")


def main():
    # for key in ARGS:
    #     ARGS[key] = rospy.get_param('~' + key, ARGS[key])
    PointingDetector()

if __name__ == '__main__':
    main()
