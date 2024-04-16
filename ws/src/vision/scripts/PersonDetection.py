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
PERCENTAGE = 0.3

class PersonDetection():

    def __init__(self):
        rospy.init_node('person_detection')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        self.check_person_service = rospy.Service(CHECK_PERSON, SetBool, self.check_person)
        self.output_img_pub = rospy.Publisher("/person_detection", Image, queue_size=1)

        self.image = None
        self.check = False
        self.model = YOLO('yolov8n.pt')
        self.output_img = []

        print("Ready")
        # rospy.spin()
        try:
            rate = rospy.Rate(60)

            while not rospy.is_shutdown():
                if len(self.output_img) != 0:
                # if VERBOSE and self.detections_frame != None:
                    cv2.imshow("Detections", self.output_img)
                    cv2.waitKey(1)

                if len(self.output_img) != 0:
                    self.output_img_pub.publish(self.bridge.cv2_to_imgmsg(self.output_img, "bgr8"))
                    
                rate.sleep()
        except KeyboardInterrupt:
            pass

        cv2.destroyAllWindows()
    
    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    
    def check_person(self, req):
        self.check = req.data
        person = False
        total = 0
        
        if self.check == False:
            return person, "Not checking for people"
        
        while not person:

            if self.image is not None:
                frame = self.image
                
                frame = self.image
                annotated_frame = frame.copy()

                width = frame.shape[1]
                results = self.model.track(frame, persist=True, tracker='bytetrack.yaml', classes=0, verbose=False) #could use botsort.yaml
                boxes = results[0].boxes.xywh.cpu().tolist()

                for box in boxes:
                    x, y, w, h = box
                    if x >= int(width*PERCENTAGE) and x <= int(width*(1-PERCENTAGE)):
                        person = True
                        total += 1
                    cv2.rectangle(annotated_frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (255, 0, 0), 2)

                    
                # # Count detected persons
                # if len(boxes) > 0:
                #     person = True
                #     total = len(boxes)

                # Display the frame
                # cv2.imshow("YOLOv8 Tracking", frame)

                # # Break the loop if 'q' is pressed
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
                self.output_img = annotated_frame

            else:
                self.check = False
                return person, "No image received"

        # Release the video capture object and close the display window
        # cv2.destroyAllWindows()
        self.check = False
        return person, f"{total} people detected"
   
        
if __name__ == '__main__':
    try:
        PersonDetection()
    except rospy.ROSInterruptException:
        pass