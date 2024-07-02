#! /usr/bin/env python3.10

import rospy
import actionlib
import frida_vision_interfaces.msg
from cv_bridge import CvBridge
import cv2

def moondream_client():
    # Creates the SimpleActionClient, passing the type of the action
    rospy.loginfo("Moondrem Client Intitialized")
    client = actionlib.SimpleActionClient('moondream', frida_vision_interfaces.msg.Moondream2Action)
    rospy.loginfo("Waiting for server")
    client.wait_for_server()
    rospy.loginfo("Server started, sending test image and 'Describe the image' prompt ")
    br = CvBridge()
    frame = cv2.imread('/moondream/assets/demo-1.jpg')
    prompt = "Describe the image"
    goal = frida_vision_interfaces.msg.Moondream2Goal(prompt=prompt,frame=br.cv2_to_imgmsg(frame))

    # Sends the goal to the action server.
    client.send_goal(goal)

    # Waits for the server to finish performing the action.
    client.wait_for_result()

    # Prints out the result of executing the action
    return client.get_result()  

if __name__ == '__main__':
    try:
        rospy.init_node('Moondream2_client')
        result = moondream_client()
        rospy.loginfo("Result: {}".format(result.response))
    except rospy.ROSInterruptException:
        print("Program interrupted before completion")
