#!/usr/bin/env python3
import rospy
import time
from PIL import Image as PILImage
import cv2
from ultralytics import YOLO

from Utils.reid_model import load_network, compare_images, extract_feature_from_img, get_structure
from Utils.pose_model import check_visibility, classify_pose, getCenterPerson
import torch.nn as nn
import torch
import tqdm
import mediapipe as mp
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from vision.srv import PersonCount, FindPerson
from vision.msg import people_count
from geometry_msgs.msg import Point 

'''
Node to count people in a room according to their poses
Subscribes to ZED CAMERA_TOPIC to recieve image
Subscribes tp START_TOPIC to start counting
Subscribes to END_TOPIC to publish results
Publishes to RESULTS_TOPIC a person_count msg
'''

CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"
START_TOPIC = "/start_counting"
END_TOPIC = "/end_counting"
RESULTS_TOPIC = "/person_counting"
FIND_TOPIC = "/find_pose"
THRESHOLD = 25

INIT_POSES = {"Sitting": 0, "Standing": 0, "Pointing right": 0, "Pointing left": 0, "Raising right hand": 0, "Raising left hand": 0, "Waving": 0, "Shirt color": ""}

ARGS = {
    "FLIP_IMAGE": False
}

class PersonCounting():

    def __init__(self):
        rospy.init_node('person_counting')
        self.bridge = CvBridge()
        self.start_service = rospy.Service(START_TOPIC, SetBool, self.toggle_start)
        self.end_service = rospy.Service(END_TOPIC, PersonCount, self.count)
        self.find_service = rospy.Service(FIND_TOPIC, FindPerson, self.find)
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        # self.end_sub = rospy.Subscriber(END_TOPIC, Bool, self.end_callback)
        self.count_pub = rospy.Publisher(RESULTS_TOPIC, people_count, queue_size=1)

        self.poses = INIT_POSES

        self.image = None
        self.start = False
        self.end = False
        self.req_pose = ""
        self.found_pose = None
        self.image_finding = []

        def loadModels():

            pbar = tqdm.tqdm(total=5, desc="Loading models")

            # Load the YOLOv8 model
            self.model = YOLO('yolov8n.pt')
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

        try:
            rate = rospy.Rate(60)

            while not rospy.is_shutdown():
                if len(self.image_finding) != 0:
                # if VERBOSE and self.detections_frame != None:
                    cv2.imshow("Finding", self.image_finding)
                    cv2.waitKey(1)

                # if len(self.image_finding) != 0:
                #     self.output_img_pub.publish(self.bridge.cv2_to_imgmsg(self.output_img, "bgr8"))
                    
                rate.sleep()
        except KeyboardInterrupt:
            pass

        cv2.destroyAllWindows()


    def toggle_start(self, req):
        self.start = req.data
        return True, "Counting started"
    
    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def start_callback(self, data):
        self.start = data

    def end_callback(self, data):
        self.end = data
        self.start = False

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

        if ARGS["FLIP_IMAGE"]:
            x = 1 - x
            y = 1 - y

        return x, y
    
    def count(self, req):
        self.start = False
        print("People detected", len(self.people_poses))

        if req.data not in self.poses:
            return "Request not found"
        
        return str(self.poses[req.data])
    
    def find(self, req):
        self.req_pose = req.request

        prev_ids = []
        while self.image is not None:
            rospy.loginfo(f"Looking for person with pose {self.req_pose}")

            # Get the frame from the camera
            frame = self.image

            found = FindPerson()
            x = 0
            y = 0
            pose = None
            
            # Get the results from the YOLOv8 model
            results = self.model.track(frame, persist=True, tracker='bytetrack.yaml', classes=0, verbose=False) #could use botsort.yaml
            
            # Get the bounding boxes and track ids
            boxes = results[0].boxes

            curr_ids = []
            for box in boxes:
                class_id = box.cls[0].item()
                print(class_id)

                if class_id not in prev_ids:
                    x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                    bbox = [x1, y1, x2, y2]
                    # bbox = box.xyxy[0].tolist()

                    # Crop the image 
                    cropped_image = frame[y1:y2, x1:x2]
                    cv2.imshow("Cropped", cropped_image)

                    # Get pose
                    pose = classify_pose(self.pose_model, cropped_image)
                    print(pose)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    if pose == None or len(pose) < 1:
                        print("No pose detected")
                        continue

                    # label = np.array2string(pose)

                    # cv2.putText(frame, label, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA) # Tag
                    
                    if self.req_pose in pose:
                        cx, cy = self.getCenter(bbox, frame)
                        
                        x = cx
                        y = cy
                        return Point(x, y, 0)
                    

                    # if self.req_pose == "":
                    #     break

                curr_ids.append(class_id)

            prev_ids = curr_ids

            self.image_finding = frame

        return Point(-1, -1, -1)

        
    def addPose(self, poses):
        for pose in poses:
            if pose not in self.poses:
                self.poses["Shirt color"] += pose + ","
            else:
                self.poses[pose] += 1

    def logPoses(self):
        print("Detected People", len(self.people_poses))

        for key, value in self.poses.items():
            print(f"{key}: {value}")

        print("-------------------------------")

    
    def run(self):
        

        # Initialize lists to store information about people
        people_tags = []
        people_ids = []
        people_features = []
        self.people_poses = []
        prev_ids = []

        rospy.loginfo("Running Person Counting")

        while rospy.is_shutdown() == False :

            if self.start == False:
                people_tags = []
                people_ids = []
                people_features = []
                self.people_poses = []
                prev_ids = []
                self.poses = INIT_POSES

            # Check if the counting has ended and publish the results
            # Check if counting is active
            elif self.start:

                # print('Counting')
                if self.image is not None:

                    # Get the frame from the camera
                    frame = self.image
                    width = frame.shape[1]
                    
                    # Get the results from the YOLOv8 model
                    results = self.model.track(frame, persist=True, tracker='bytetrack.yaml', classes=0, verbose=False) #could use botsort.yaml
                    
                    # Get the bounding boxes and track ids
                    boxes = results[0].boxes.xywh.cpu().tolist()
                    track_ids = []

                    try:
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                    except Exception as e:
                        track_ids = []

                    false_detections = []

                    # Check if there is a new id
                    for (box, track_id) in zip(boxes, track_ids):
                        if track_id not in prev_ids:

                            # Get bbox
                            x = int(box[0])
                            y = int(box[1])
                            w = int(box[2])
                            h = int(box[3]) 
                            x1 = int(x - w / 2)
                            y1 = int(y - h / 2)
                            x2 = int(x + w / 2)
                            y2 = int(y + h / 2)

                            # Crop the image 
                            cropped_image = frame[y1:y2, x1:x2]
                            # cv2.imshow("Cropped", cropped_image)
                            pil_image = PILImage.fromarray(cropped_image)
                            person = check_visibility(self.pose_model,cropped_image)

                            # Check if the person is visible enough (chest) and not too close to the edges of the image
                            if not person or x1 <= THRESHOLD or x2 >= width - THRESHOLD:
                                false_detections.append(track_id)
                                continue

                            # Get feature for REID model
                            with torch.no_grad():
                                new_feature = extract_feature_from_img(pil_image, self.model_reid)
                            flag = False

                            # Get pose
                            pose = classify_pose(self.pose_model, cropped_image)

                            # Check if there is a match with seen people
                            for i, person_feature in enumerate(people_features):

                                #Compare features
                                match = compare_images(person_feature, new_feature)

                                # If there is a match and the person matched is not currently in the frame (shouldnt be two people with the same id in the same frame)
                                if match and people_ids[i] not in track_ids:

                                    # Update id to the id assigned by yolo and pose
                                    people_ids[i] = track_id
                                    self.people_poses[i] = pose
                                    flag = True
                                    break
                            
                            # If there is no match and the id is not already saved (New person detected)
                            if not flag and track_id not in people_ids:
                                print("New person detected")
                                people_ids.append(track_id)
                                people_tags.append(f"Person {len(people_ids)}")
                                people_features.append(new_feature)
                                self.people_poses.append(pose)

                                
                                    
                                    # print("Pose detected", pose)
                                    # print(pose)
                                    
                                self.addPose(pose)



                    # print(track_ids)
                    # print(people_tags)
                    # print(self.people_poses)
                    self.logPoses()
                    prev_ids = []

                    # Draw results and update prev detections
                    for (i, track_id) in enumerate(track_ids):
                        box = boxes[i]
                        if track_id in false_detections:
                            continue

                        prev_ids.append(track_id)

                        x = int(box[0])
                        y = int(box[1])
                        w = int(box[2])
                        h = int(box[3]) 

                        id = people_ids.index(track_id)
                        name = people_tags[id]

                        cv2.rectangle(frame, (int(x - w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255, 0, 0), 2)
                        cv2.putText(frame, name, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA) # Tag
                        cv2.putText(frame, str(track_id), (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA) # Yolo ID

                    
                    # Display the annotated frame
                    cv2.imshow("YOLOv8 Tracking", frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        # Release the video capture object and close the display window
        cv2.destroyAllWindows()

# Run node

if __name__ == "__main__":
    try: 
        for key in ARGS:
            ARGS[key] = rospy.get_param('~' + key, ARGS[key])

        PersonCounting()
    except rospy.ROSInterruptException:
        pass