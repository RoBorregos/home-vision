#!/usr/bin/env python3
import cv2
from ultralytics import YOLO

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool

'''
Node to detect people
Subscribes to ZED CAMERA_TOPIC to recieve image
Service to check if a person is detected
    Returns true when person or people detected and 
    string with number of people detected
'''


CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"
CHECK_PERSON = "/check_person"

class PersonDetection():

    def __init__(self):
        rospy.init_node('person_detection')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        self.check_person_service = rospy.Service(CHECK_PERSON, SetBool, self.check_person)
        self.image = None
        self.check = False
        self.model = YOLO('yolov8n.pt')

        print("Ready")
        rospy.spin()
    
    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    
    def check_person(self, req):
        self.check = req.data
        person = False
        total = 0
        
        while not person:

            if self.image is not None:
                frame = self.image
                
                frame = self.image
                results = self.model.track(frame, persist=True, show=True, tracker='bytetrack.yaml', classes=0, verbose=False) #could use botsort.yaml
                boxes = results[0].boxes.xywh.cpu().tolist()

                # Count detected persons
                if len(boxes) > 0:
                    person = True
                    total = len(boxes)

                # Display the frame
                cv2.imshow("YOLOv8 Tracking", frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Release the video capture object and close the display window
        cv2.destroyAllWindows()
            
        return person, f"{total} people detected"
   
        
if __name__ == '__main__':
    try:
        PersonDetection()
    except rospy.ROSInterruptException:
        pass