#!/usr/bin/env python3
import rospy
import actionlib
from frida_vision_interfaces.msg import DetectPointingObjectAction, DetectPointingObjectGoal, DetectPointingObjectResult, DetectPointingObjectFeedback

class DetectPointingObjectCaller:
    def __init__(self):
        rospy.loginfo("Connecting to Detect Pointing Object Server")
        self.client = actionlib.SimpleActionClient('detectPointingObject', DetectPointingObjectAction)
        self.client.wait_for_server()
        rospy.loginfo("Connected to Detect Pointing Object Server")

    def execute(self):
        goal = DetectPointingObjectGoal(waiting_time=5)
        
        self.client.send_goal(goal)
        print("Waiting result")
        self.client.wait_for_result()
        return self.client.get_result()

if __name__ == '__main__':
    try:
        rospy.init_node('detect_pointing_object_caller')
        caller = DetectPointingObjectCaller()
        result = caller.execute()
        rospy.loginfo(f"Result: {result}")
        rospy.loginfo(f"Pose Result: {result.point3D}")
    except rospy.ROSInterruptException as e:
        rospy.logerr(f'Error: {e}')