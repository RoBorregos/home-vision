#! /usr/bin/env python3

import rospy
import actionlib
from frida_vision_interfaces.msg import MoondreamFromCameraAction, MoondreamFromCameraGoal, MoondreamFromCameraResult, MoondreamFromCameraFeedback
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image

ACTION_NAME = "/vision/moondream_from_camera"
CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"
X1, Y1, X2, Y2 = 0.0, 0.0, 0.5, 0.5

class MoondreamCameraActionClient:
    def __init__(self):
        rospy.init_node('moondream_camera_client')
        self.as_client = actionlib.SimpleActionClient(ACTION_NAME, MoondreamFromCameraAction)
        print("Waiting for server")
        self.as_client.wait_for_server()
        goal = MoondreamFromCameraGoal(camera_topic=CAMERA_TOPIC, roi_x1=X1, roi_y1=Y1, roi_x2=X2, roi_y2=Y2, prompt="Describe the image")
        print("Moondream Camera Action Client Started, sent goal: ", goal)
        self.as_client.send_goal(goal)
        self.as_client.wait_for_result()
        result = self.as_client.get_result()
        
        rospy.loginfo(f"Moondream From Camera Client got result: {result}")
        
if __name__ == '__main__':
    server = MoondreamCameraActionClient()