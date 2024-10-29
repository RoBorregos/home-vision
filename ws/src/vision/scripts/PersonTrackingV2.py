#!/usr/bin/env python3
import rospy
import time
from PIL import Image as PILImage
import cv2
from ultralytics import YOLO
# from ReID import reid_model
#dlib

from Utils.reid_model import load_network, compare_images, extract_feature_from_img, get_structure
from Utils.pose_model import check_visibility, getCenterPerson
import torch.nn as nn
import torch
import tqdm
import mediapipe as mp

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Point
from vision.msg import img, img_list, target
from std_srvs.srv import SetBool
import imutils
import numpy as np


import face_recognition


'''
Script to track people
Service to follow the person with largest bounding box when called
Publishes center of bbox of person tracked 
'''

ARGS = {
    "FLIP_IMAGE": False
}

# Server topics
CHANGE_TRACKING_TOPIC = "/vision/change_person_tracker_state"

# Subscribe topics
CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"

# Publish topics
DETECTION_TOPIC = "/vision/person_detection"
IMAGE_PUB_TOPIC = "/vision/img_tracking"

# Constants
THRESHOLD = 15
EMB_THRESHOLD = 10


class PersonTracking():

    def __init__(self):
        rospy.init_node('person_tracking')
        self.bridge = CvBridge()

        for key in ARGS:
            print(key)
            ARGS[key] = rospy.get_param(key, False)
            print(rospy.get_param(key, False))
            print(ARGS[key])

        self.track_service = rospy.Service(CHANGE_TRACKING_TOPIC, SetBool, self.toggle_tracking)
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        self.detection_pub = rospy.Publisher(DETECTION_TOPIC, Point, queue_size=1)
        self.image_pub = rospy.Publisher(IMAGE_PUB_TOPIC, Image, queue_size=1)

        rospy.loginfo("Flipped image: " + str(ARGS["FLIP_IMAGE"]))
        self.model = YOLO('yolov8n.pt')
        self.image = None
        self.track = False

        def loadModels():
            pbar = tqdm.tqdm(total=5, desc="Loading models")

            # Load the YOLOv8 model
            pbar.update(1)

            # Load media pipe model
            self.pose_model = mp.solutions.pose.Pose(min_detection_confidence=0.8) 
            pbar.update(1)

            # Load the ReID model
            structure = get_structure()
            pbar.update(1)
            self.model_reid = load_network(structure)
            pbar.update(1)
            self.model_reid.classifier.classifier = nn.Sequential()
            pbar.update(1)
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                self.model_reid = self.model_reid.cuda()
            pbar.close()

        loadModels()

        rospy.loginfo("Person Tracking Ready")
        self.run()
    
    def toggle_tracking(self, req):
        self.track = req.data
        rospy.loginfo("Tracking is now " + ("enabled" if self.track else "disabled"))
        return True, "Tracking is now " + ("enabled" if self.track else "disabled")

    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def getCenter(self, box, frame):
        x1, y1, x2, y2 = [int(i) for i in box]

        crop = frame[y1:y2, x1:x2]
        x, y = getCenterPerson(self.pose_model, crop)

        if x == None or y == None:
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
        
        else:
            x = x1 + x
            y = y1 + y

        width = frame.shape[1]
        height = frame.shape[0]

        if ARGS["FLIP_IMAGE"]:
            x /= width
            y /= height
            x = 1 - x
            y = 1 - y
            x *= width
            y*= height

        return x, y
            

    def run(self):
        # Initialize lists to store information about people
        people_tags = []
        people_ids = []
        people_features = []
        prev_ids = []
        track_person = ""
        emb_count = []

        rospy.loginfo("Running Person Tracking")

        while rospy.is_shutdown() == False :
            if self.track is False:
                people_tags = []
                people_ids = []
                people_features = []
                prev_ids = []
                track_person = ""
                emb_count = {}
                
            
            else:
                if self.image is not None:

                    # Get the frame from the camera
                    frame = self.image
                    if ARGS["FLIP_IMAGE"]:
                        frame = imutils.rotate(frame, 180)

                    width = frame.shape[1]
                    
                    # Get the results from the YOLOv8 model
                    results = self.model.track(frame, persist=True, tracker='bytetrack.yaml', classes=0, verbose=False) #could use botsort.yaml
                    
                    # Get face locations and encodings
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    face_locations = face_recognition.face_locations(frame)

                    

                    # Get the bounding boxes and track ids
                    boxes = results[0].boxes.xyxy.cpu().tolist()
                    track_ids = []

                    try:
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                    except Exception as e:
                        track_ids = []

                    false_detections = []

                    # Check if there is a new id
                    for (box, track_id) in zip(boxes, track_ids):
                        # if track_id not in emb_count:
                        #     emb_count[track_id] = 0

                        # emb_count[track_id] += 1
                        new_feature = None

                        if track_id not in prev_ids:

                            # Get bbox
                            
                            x1, y1, x2, y2 = [int(i) for i in box]
                               

                            # Crop the image 
                            cropped_image = frame[y1:y2, x1:x2]
                            pil_image = PILImage.fromarray(cropped_image)
                            person = check_visibility(self.pose_model,cropped_image)

                            if not person or x1 <= THRESHOLD or x2 >= width - THRESHOLD:
                                false_detections.append(track_id)
                                continue

                            # Get feature
                            with torch.no_grad():
                                new_feature = extract_feature_from_img(pil_image, self.model_reid)
                            flag = False

                            # Check if there is a match with seen people
                            for i, person_feature in enumerate(people_features):

                                #Compare features
                                match = compare_images(person_feature, new_feature)

                                # If there is a match and the person matched is not currently in the frame (shouldnt be two people with the same id in the same frame)
                                if match and people_ids[i] not in track_ids:

                                    # Update id to the id assigned by yolo
                                    people_ids[i] = track_id
                                    flag = True
                                    break
                            
                            # If there is no match and the id is not already saved
                            if not flag and track_id not in people_ids:
                                rospy.loginfo("New person detected")
                                people_ids.append(track_id)
                                people_tags.append(f"Person {len(people_ids)}")
                                # emb_count[track_id] += 1 

                       
                        # if emb_count[track_id] % EMB_THRESHOLD == 0:
                        #     if new_feature is None:
                        #         with torch.no_grad():
                        #             new_feature = extract_feature_from_img(pil_image, self.model_reid)
                        #     people_features[track_id] = np.mean([people_features[track_id], new_feature], axis=0)
                            
                    prev_ids = []
                    
                    # Draw results
                    max_area = 0
                    index = 0
                    cx = -1
                    cy = -1

                    for (i, track_id) in enumerate(track_ids):
                        box = boxes[i]
                        if track_id in false_detections:
                            continue

                        prev_ids.append(track_id)

                        x1, y1, x2, y2 = [int(i) for i in box]
                            
                        x = int((x1 + x2) / 2)
                        y = int((y1 + y2) / 2)
                        w = x2 - x1
                        h = y2 - y1

                        tag_index = people_ids.index(track_id)
                        
                        if track_person == people_tags[tag_index]:  
                            cx, cy = self.getCenter(box, frame)
 
                        
                        if w*h > max_area:
                            max_area = w*h
                            index = i

                        id = people_ids.index(track_id)
                        name = people_tags[id]

                        cv2.rectangle(frame, (int(x - w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
                        cv2.putText(frame, name, (int(x - w/2) + 10, int(y-h/2) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) # Tag

                    
                    
                    if self.track and track_person == "" and max_area != 0:
                        index = people_ids.index(track_ids[index])
                        track_person = people_tags[index]


                    if self.track and cx != -1 and cy != -1:
                        msg = Point()
                        msg.x = cx
                        msg.y = cy
                        msg.z = 0
                        self.detection_pub.publish(msg)

                    
                    # Display the annotated frame
                    cv2.imshow("Person Tracking", frame)
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
                    
                    rospy.loginfo("Tracking " + track_person)   

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    rospy.loginfo("No image")


        # Release the video capture object and close the display window
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try: 
        PersonTracking()
    except rospy.ROSInterruptException:
        pass