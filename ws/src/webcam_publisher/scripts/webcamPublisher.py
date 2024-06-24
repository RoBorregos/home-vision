#! /usr/bin/env python3

# Publish webcam images to a ROS topic

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

class WebcamPublisher(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)
        self.pub = rospy.Publisher("webcam_image", Image, queue_size=10)
        rospy.loginfo("Webcam Publisher Initialized")
        self.publish()

    def publish(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                img = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.pub.publish(img)
            time.sleep(0.1)

if __name__ == '__main__':
    rospy.init_node('webcam_publisher')
    WebcamPublisher()
    rospy.spin()