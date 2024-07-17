#! /usr/bin/env python3

import rospy
import actionlib
from frida_vision_interfaces.msg import Moondream2Action, Moondream2Goal, Moondream2Result, Moondream2Feedback
from frida_vision_interfaces.msg import MoondreamFromCameraAction, MoondreamFromCameraGoal, MoondreamFromCameraResult, MoondreamFromCameraFeedback
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image

ACTION_NAME = "/vision/moondream_from_camera"
MOONDREAM2_AS = "moondream"

class MoondreamCameraAction:
    def __init__(self):
        rospy.init_node('moondream_camera')
        self.moondream_client = actionlib.SimpleActionClient(MOONDREAM2_AS, Moondream2Action)
        rospy.loginfo("Waiting for Moondream2 Server")
        self.moondream_client.wait_for_server()
        self._action_name = ACTION_NAME
        self._as = actionlib.SimpleActionServer(self._action_name, MoondreamFromCameraAction, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()
        rospy.loginfo("Moondream Camera Action Server Started")
        
    def execute_cb(self, goal):
        camera_topic = goal.camera_topic
        x1, y1, x2, y2 = goal.roi_x1, goal.roi_y1, goal.roi_x2, goal.roi_y2
        
        rospy.loginfo(f"MoonDream Camera Action Server: Executing, resolving frame with prompt {goal.prompt} and ROI ({x1}, {y1}), ({x2}, {y2})")
        
        image = rospy.wait_for_message(camera_topic, Image)
        image = CvBridge().imgmsg_to_cv2(image, "bgr8")
        
        if x1 != 0 or y1 != 0 or x2 != 0 or y2 != 0:
            image = image[int(y1*image.shape[0]):int(y2*image.shape[0]), int(x1*image.shape[1]):int(x2*image.shape[1])]
        
        prompt = goal.prompt
        
        cv2.imwrite("moondream_camera_service.jpg", image)
        moondream_goal = Moondream2Goal(prompt=prompt, frame=CvBridge().cv2_to_imgmsg(image))
        self.moondream_client.send_goal(moondream_goal)
        self.moondream_client.wait_for_result()
        result = self.moondream_client.get_result()
        
        rospy.loginfo(f"Moondream Camera Client got result: {result}")
        self._as.set_succeeded(MoondreamFromCameraResult(response=result.response))
    
if __name__ == '__main__':
    server = MoondreamCameraAction()
    rospy.spin()