#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import pathlib

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from frida_vision_interfaces.srv import FindSeat
import numpy as np
import queue

'''
Node to find a seat for a person in a couch or chair
'''

CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"
FIND_TOPIC = "/find_seat"
OUTPUT_IMAGE = "/seat_detection"
MAX_DEGREE = 30

model_location = str(pathlib.Path(__file__).parent) + "/Utils/yolov8n.pt"

class SeatFinding():

    def __init__(self):
        rospy.init_node('seat_finding')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        self.find_seat_service = rospy.Service(FIND_TOPIC, FindSeat, self.find_seat)
        self.output_img_pub = rospy.Publisher(OUTPUT_IMAGE, Image, queue_size=1)
        self.model = YOLO(model_location)
        self.image = None
        self.output_img = []

        print("Seat Finding Ready")
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
        # path = str(pathlib.Path(__file__).parent) + "/test4.png"
        # self.image = cv2.imread(path)

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
                    # print(label)

            for chair in chairs:
                occupied = False
                xmin = chair["bbox"][0]
                xmax = chair["bbox"][2]

                for person in people:
                    center_x = (person["bbox"][0] + person["bbox"][2]) / 2

                    if center_x >= xmin and center_x <= xmax:
                        occupied = True
                        print("Occupied")
                        break

                if not occupied:
                    output = (chair["bbox"][0] + chair["bbox"][2]) / 2
                    print("Chair found", output)
                    cv2.rectangle(self.output_img, (chair["bbox"][0], chair["bbox"][1]), (chair["bbox"][2], chair["bbox"][3]), (0, 0, 255), 2)
                    return self.getAngle(output, frame.shape[1])

            print("No chair found")

            available_spaces = queue.PriorityQueue()

            for couch in couches:
                couch_left = couch["bbox"][0]
                couch_right = couch["bbox"][2]

                space = np.zeros(frame.shape[1], dtype=int)
                
                # Fill the space with 1 if there is a person
                for person in people:
                    xmin = person["bbox"][0]
                    xmax = person["bbox"][2]

                    space[xmin:xmax+1] = 1

                # print('space', space)
                
                print('people', len(people))

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

            print(f"Found {len(couches)} couches")

            # if people_sitting == 2:
            print("qsize", available_spaces.qsize())
            if available_spaces.qsize() > 0:
                max_space, left, right = available_spaces.get()
                output = (left + right) / 2
                cv2.rectangle(self.output_img, (left, 0), (right, frame.shape[0]), (0, 0, 255), 2)
                print("Space found", output)
                return self.getAngle(output, frame.shape[1])
            
            else:
                print("No couch or chair found")
                return -1
            
            # else:

        else:
            return -1 # No seat found
            


if __name__ == "__main__":
    try: 
        SeatFinding()
    except rospy.ROSInterruptException:
        pass