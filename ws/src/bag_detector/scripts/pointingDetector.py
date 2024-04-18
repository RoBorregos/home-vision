#!/usr/bin/env python3
import rospy
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
from math import acos, degrees
from enum import Enum
import cv2
import time as timelib

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int32, Bool
from frida_vision_interfaces.srv import Pointing
# from bag_detector.srv import Pointing


POINTING_TOPIC = "/detectios/pointing"
CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"
RES_TOPIC = "/res"

class Direction(Enum):
    NOT_POINTING = 0
    RIGHT = 1
    LEFT = 2

labels = ["Not pointing", "Left", "Right"]

class PointingDetector:
    def __init__(self):
        rospy.init_node('pointing_detector')
        self.bridge = CvBridge()
        self.pointing_service = rospy.Service(POINTING_TOPIC, Pointing, self.pointing)
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)

        self.results_pub = rospy.Publisher(RES_TOPIC, Int32, queue_size=1)
        self.image_pointing = []
        self.annotated = []

        def load_models():
            self.model = YOLO('yolov8n.pt')
            self.pose_model = mp.solutions.pose.Pose(min_detection_confidence=0.8)

        load_models()
        self.image = None
        self.start = False

        rospy.loginfo("Running pointing node")
        # self.run()

        try:
            rate = rospy.Rate(60)

            while not rospy.is_shutdown():
                # if len(self.image_pointing) != 0:
                # # if VERBOSE and self.detections_frame != None:
                #     # cv2.imshow("Pose analysis", self.image_pointing)
                #     # cv2.waitKey(1)

                if len(self.annotated) != 0:
                    cv2.imshow("Results", self.annotated)
                    cv2.waitKey(1)

                # if len(self.image_finding) != 0:
                #     self.output_img_pub.publish(self.bridge.cv2_to_imgmsg(self.output_img, "bgr8"))
                    
                rate.sleep()
        except KeyboardInterrupt:
            pass

        cv2.destroyAllWindows()


    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def pointing(self, req):
        self.start = req.active
        time = req.seconds
        ans = self.run(time)
        return ans
    
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
        results = self.pose_model.process(image)
        direction = 0

        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            shoulder_right = landmarks[12]
            shoulder_left = landmarks[11]
            wrist_left = landmarks[15]
            wrist_right = landmarks[16]
            index_right = landmarks[20]
            hip_right = landmarks[24]
            elbow_right = landmarks[14]

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            self.image_pointing = image

            x_center = (shoulder_right.x + shoulder_left.x) / 2

            angle_body = self.getAngle(hip_right, shoulder_right, elbow_right)
            arm_angle = self.getAngle(shoulder_right, elbow_right, wrist_right)

            if angle_body < 33:
                if arm_angle > 150 and arm_angle < 200:
                    direction = 0

                else:
                    if index_right.x > elbow_right.x:
                        direction = 1
                    else:
                        direction = 2
                
            else:
                if index_right.x > shoulder_right.x:
                    direction = 1

                else:
                    direction = 2

        return (direction)



            # y_center = (shoulder_right.y + shoulder_left.y) / 2

    def run(self, time):
        start_time = timelib.time()
        end_time = start_time + (time)
        self.start = True

        avgs = [0,0,0]
        pointing_direction = 0

        while timelib.time() < end_time:

            if self.image is not None:

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

                            cv2.rectangle(frame, (x1, int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                x1, y1, x2, y2 = max_bbox
                crop = frame[y1:y2, x1:x2]
                pointing_direction = (self.getDirection(crop))

                print(labels[pointing_direction])
                
                cv2.putText(frame, labels[pointing_direction], (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) # Tag

                self.results_pub.publish(pointing_direction)
                self.annotated = frame
                avgs[pointing_direction] += 1
                

            else:
                rospy.loginfo("No image provided")
                    
        self.annotated = []
        self.image_pointing = []
        cv2.destroyAllWindows()

        rospy.loginfo(f"Result: {labels[pointing_direction]}")

        return pointing_direction




def main():
    # for key in ARGS:
    #     ARGS[key] = rospy.get_param('~' + key, ARGS[key])
    PointingDetector()

if __name__ == '__main__':
    main()
