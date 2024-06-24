#!/usr/bin/env python3

"""
Node to detect people
Subscribes to ZED CAMERA_TOPIC to recieve image
Service to check if a person is detected
    Returns true when person or people detected and 
    string with number of people detected
"""

import cv2
from ultralytics import YOLO
import pathlib

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from frida_vision_interfaces.srv import FindSeat
# from vision.srv import FindSeat
from std_srvs.srv import SetBool

import numpy as np
import queue

# Server topics
CHECK_PERSON = "/vision/check_person"
FIND_TOPIC = "/vision/find_seat"

# Subscribe topics
CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"

# Publish topics
IMAGE_TOPIC = "/vision/img_person_detection"

# Constants
MODEL_LOCATION = str(pathlib.Path(__file__).parent) + "/Utils/yolov8n.pt"
PERCENTAGE = 0.3
MAX_DEGREE = 30

class ReceptionistCommands():

    def __init__(self):
        rospy.init_node('receptionist_commands')
        self.bridge = CvBridge()

        self.check_person_service = rospy.Service(CHECK_PERSON, SetBool, self.check_person)
        self.find_seat_service = rospy.Service(FIND_TOPIC, FindSeat, self.find_seat)
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        self.output_img_pub = rospy.Publisher(IMAGE_TOPIC, Image, queue_size=1)

        self.image = None
        self.check = False
        self.model = YOLO(MODEL_LOCATION)
        self.output_img = []

        rospy.loginfo("Person Detection Ready")

        try:
            rate = rospy.Rate(60)

            while not rospy.is_shutdown():
                if len(self.output_img) != 0:
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


    def getAngle(self, x, width):
        diff = x - (width / 2)
        move = diff * MAX_DEGREE / (width / 2)
        return move
    
    
    def find_seat(self, req):

        if self.image is not None:
            frame = self.image
            self.output_img = frame
            
            results = self.model(frame, verbose=False, classes=[0,56,57])
            output = 0

            people = []
            chairs = []
            couches = []

            # Get detections
            for out in results:
                for box in out.boxes:
                    x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                    class_id = box.cls[0].item()
                    label = self.model.names[class_id]
                    bbox = (x1, y1, x2, y2)

                    if class_id == 0:
                        people.append({"bbox": bbox, "label": label, "class": class_id})
                    
                    elif class_id == 56:
                        chairs.append({"bbox": bbox, "label": label, "class": class_id})

                    elif class_id == 57:
                        couches.append({"bbox": bbox, "label": label, "class": class_id})

                    cv2.rectangle(self.output_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(self.output_img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            chair_q = queue.PriorityQueue()

            # Check if there are chairs available
            for chair in chairs:
                occupied = False
                xmin = chair["bbox"][0]
                xmax = chair["bbox"][2]
                y_center_chair = (chair["bbox"][1] + chair["bbox"][3]) / 2
                cv2.circle(self.output_img, (int(xmin), int(y_center_chair)), 5, (0, 255, 255), -1)


                for person in people:
                    center_x = (person["bbox"][0] + person["bbox"][2]) / 2
                    person_y = person["bbox"][3]

                    if center_x >= xmin and center_x <= xmax and person_y > y_center_chair:
                        occupied = True
                        cv2.circle(self.output_img, (int(center_x), int(person_y)), 5, (0, 0, 255), -1)
                        break

                if not occupied:
                    area = xmax - xmin
                    output = (chair["bbox"][0] + chair["bbox"][2]) / 2
                    chair_q.put((-1*area, output, chair["bbox"][0],chair["bbox"][1],chair["bbox"][2],chair["bbox"][3]))
                else:
                    cv2.rectangle(self.output_img, (xmin, chair["bbox"][1]), (xmax, chair["bbox"][3]), (0, 0, 255), 2)
                    
                
            if len(chairs) != 0:
                space, output, a,b,c,d = chair_q.get()
                cv2.rectangle(self.output_img, (a, b), (c, d), (255, 255, 0), 2)
                rospy.loginfo(f"Chair found: {output}")
                return self.getAngle(output, frame.shape[1])

            rospy.loginfo("No chair found")

            available_spaces = queue.PriorityQueue()

            # Check if there are couch spaces available
            for couch in couches:
                couch_left = couch["bbox"][0]
                couch_right = couch["bbox"][2]
                space = np.zeros(frame.shape[1], dtype=int)
                
                # Fill the space with 1 if there is a person
                for person in people:
                    xmin = person["bbox"][0]
                    xmax = person["bbox"][2]

                    space[xmin:xmax+1] = 1

                left = couch_left
                space[couch_left] = 0
                space[couch_right] = 0

                for i in range(couch_left, couch_right):
                    if space[i] == 0:
                        if left is None:
                            left = i
                        
                    else:
                        if left is not None:
                            available_spaces.put((-1*(i - left), left, i))
                            left = None
                    

                if left is not None:
                    available_spaces.put((-1*(couch_right - left), left, couch_right))

            if available_spaces.qsize() > 0:
                max_space, left, right = available_spaces.get()
                output = (left + right) / 2
                cv2.rectangle(self.output_img, (left, 0), (right, frame.shape[0]), (0, 0, 255), 2)
                rospy.loginfo(f"Space found: {output}")
                return self.getAngle(output, frame.shape[1])
            
            else:
                rospy.loginfo("No couch or chair found")
                return -1

        else:
            return -1 # No seat found
    

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

                self.output_img = annotated_frame

            else:
                self.check = False
                return person, "No image received"

        rospy.loginfo(f"{total} people detected")
        self.check = False
        return person, f"{total} people detected"
   
        
if __name__ == '__main__':
    try:
        ReceptionistCommands()
    except rospy.ROSInterruptException:
        pass