#! /usr/bin/env python3 

# detects the pointing direction of the user's hand
import rospy
import actionlib
from frida_vision_interfaces.msg import DetectPointingObjectAction, DetectPointingObjectGoal, DetectPointingObjectResult, DetectPointingObjectFeedback
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np
import mediapipe as mp

USE_RIGHT_HAND = True
USE_LEFT_HAND = False

class DetectPointingObjectServer(object):
        
        def __init__(self):
            rospy.loginfo("Initializing Detect Pointing Object Server")
            
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose()
            
            self.VISUALIZE = True
            self.RIGHT_SHOULDER = 12
            self.RIGHT_HAND = 20
            self.LEFT_SHOULDER = 11
            self.LEFT_HAND = 19
            
            self.object_centroids = {
                0: (0.5, 0.9),
                1: (0.1, 0.9),
                2: (0.9, 0.9),
            }
            
            self.bridge = CvBridge()
            self.image_sub = rospy.Subscriber("webcam_image", Image, self.image_callback)
            self.visualizer_pub = rospy.Publisher("pointing_direction", Image, queue_size=10)
            
            self.server = actionlib.SimpleActionServer('detectPointingObject', DetectPointingObjectAction, execute_cb=self.execute_cb, auto_start=False)
            self.server.start()
            rospy.loginfo("Detect Pointing Object Server Initialized")
            
        def execute_cb(self, goal):
            rospy.loginfo("Detect Pointing Object Server Received Goal")
            result = DetectPointingObjectResult()
            feedback = DetectPointingObjectFeedback()
            
            self.server.set_succeeded(result)
            rospy.loginfo("Detect Pointing Object Server Finished")
        
        def image_callback(self, data):
            img = self.bridge.imgmsg_to_cv2(data)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            visualize_img = img.copy()
            img.flags.writeable = False
            results = self.pose.process(img)
            
            # draw points
            ONLY_SHOULDER_AND_HANDS = True
            if results.pose_landmarks:
                for i, point in enumerate(results.pose_landmarks.landmark):
                    if self.VISUALIZE and (not ONLY_SHOULDER_AND_HANDS or i == self.RIGHT_SHOULDER or i == self.RIGHT_HAND or i == self.LEFT_SHOULDER or i == self.LEFT_HAND):
                        visualize_img = cv2.circle(visualize_img, (int(point.x * img.shape[1]), int(point.y * img.shape[0])), 5, (0, 255, 0), -1)
                        cv2.putText(visualize_img, str(i), (int(point.x * img.shape[1]), int(point.y * img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            closest_object = None
            closest_distance = 100000
            
            # get angle between finger tip and hand base
            if results.pose_landmarks:
                
                finger_tip_left = None
                hand_base_left = None
                finger_tip_right = None
                hand_base_right = None
                
                for i, point in enumerate(results.pose_landmarks.landmark):
                    if i == self.RIGHT_HAND and self.check_visible(point, img):
                        finger_tip_right = point
                    if i == self.RIGHT_SHOULDER and self.check_visible(point, img):
                        hand_base_right = point
                    if i == self.LEFT_HAND and self.check_visible(point, img):
                        finger_tip_left = point
                    if i == self.LEFT_SHOULDER and self.check_visible(point, img):
                        hand_base_left = point
                        
                
                
                print("-------------------")
                # generate the linear equations for the points
                
                
                if (finger_tip_right is not None and hand_base_right is not None) and USE_RIGHT_HAND:
                    print(f"Right Hand: {finger_tip_right.x}, {finger_tip_right.y}")
                    print(f"Right Shoulder: {hand_base_right.x}, {hand_base_right.y}")
                    right_m = (finger_tip_right.y - hand_base_right.y) / (finger_tip_right.x - hand_base_right.x)
                    right_intercept = finger_tip_right.y - right_m * finger_tip_right.x
                    right_line_start = (0, int(right_intercept * img.shape[0]))
                    right_line_end = (img.shape[1], int((right_m * 1 + right_intercept) * img.shape[0]))
                    cv2.line(visualize_img, right_line_start, right_line_end, (0, 255, 0), 2)
                    closest_object, closest_distance, visualize_img = self.check_closest_object(right_m, right_intercept, visualize_img)
                if (finger_tip_left is not None and hand_base_left is not None) and USE_LEFT_HAND:
                    left_m = (finger_tip_left.y - hand_base_left.y) / (finger_tip_left.x - hand_base_left.x)
                    print(f"Left Hand: {finger_tip_left.x}, {finger_tip_left.y}")
                    print(f"Left Shoulder: {hand_base_left.x}, {hand_base_left.y}")
                    left_intercept = finger_tip_left.y - left_m * finger_tip_left.x
                    left_line_start = (0, int(left_intercept * img.shape[0]))
                    left_line_end = (img.shape[1], int((left_m * 1 + left_intercept) * img.shape[0]))
                    cv2.line(visualize_img, left_line_start, left_line_end, (0, 255, 0), 2)
                    closest_object, closest_distance, visualize_img = self.check_closest_object(left_m, left_intercept, visualize_img)
                    
                if closest_object is not None:
                    print(f"Closest Object: {closest_object}, Distance: {closest_distance}")
                    
                
            
            if self.VISUALIZE:
                # visualize points
                for i, point in enumerate(self.object_centroids):
                    color = (0, 255, 0) if i == closest_object else (0, 0, 0)
                    visualize_img = cv2.circle(visualize_img, (int(self.object_centroids[point][0] * img.shape[1]), int(self.object_centroids[point][1] * img.shape[0])), 5, color, -1)
                    visualize_img = cv2.putText(visualize_img, str(i), (int(self.object_centroids[point][0] * img.shape[1]), int(self.object_centroids[point][1] * img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                visualize_img = cv2.cvtColor(visualize_img, cv2.COLOR_BGR2RGB)
                img_msg = self.bridge.cv2_to_imgmsg(visualize_img, encoding="bgr8")
                self.visualizer_pub.publish(img_msg)
            
        def check_visible(self, point, img):
            if point.x < 0 or point.x > 1 or point.y < 0 or point.y > 1:
                return False
            return True

        def check_closest_object(self, m, intercept, visualize_img):
            closest_object = None
            closest_distance = 100000
            for i, centroid in self.object_centroids.items():
                # visualize_img = self.draw_orthogonal_distance(m, intercept, centroid, visualize_img)
                distance = abs(centroid[1] - m * centroid[0] - intercept) / np.sqrt(m**2 + 1)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_object = i
            print(f"Closest Object: {closest_object}, Distance: {closest_distance}")
            return closest_object, closest_distance, visualize_img

        def draw_orthogonal_distance(self, m, intercept, centroid, img):
            # draw orthogonal line
            orthogonal_m = -1 / m
            orthogonal_intercept = centroid[1] - orthogonal_m * centroid[0]
            orthogonal_line_start = (0, int(orthogonal_intercept * img.shape[0]))
            orthogonal_line_end = (img.shape[1], int((orthogonal_m * 1 + orthogonal_intercept) * img.shape[0]))
            cv2.line(img, orthogonal_line_start, orthogonal_line_end, (0, 0, 255), 2)
            return img
        

if __name__ == '__main__':
    rospy.init_node('DetectPointingObjectServer')
    DetectPointingObjectServer()
    rospy.spin()