#!/usr/bin/env python3
import rospy
import time
from PIL import Image as PILImage
import cv2
from ultralytics import YOLO
# from ReID import reid_model
from Utils.pose_model import classify_pose
from Utils.reid_model import check_visibility, load_network, compare_images, extract_feature_from_img, get_structure
import torch.nn as nn
import torch
import tqdm
import mediapipe as mp

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Point
from vision.msg import img, img_list, target
from vision.srv import PersonCount
from std_srvs.srv import SetBool
import imutils
import threading

ARGS = {
    "FLIP_IMAGE": False
}

CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"
TRACKING_TOPIC = "/track_person"
DETECTION_TOPIC = "/person_detection"
CHANGE_TRACKING_TOPIC = "/change_person_tracker_state"
START_TOPIC = "/start_counting"
END_TOPIC = "/end_counting"
RESULTS_TOPIC = "/person_counting"

THRESHOLD = 15

# Script to track people
# Subscribes to bool TRACKING_TOPIC to follow the person with largest bounding box
# Subscribes to ZED CAMERA_TOPIC to recieve image
# Publishes center of bbox of person tracked (-1 if not found) DETECTION_TOPIC


class PersonTracking():

    def __init__(self):
        rospy.init_node('person_tracking')
        self.bridge = CvBridge()
        self.track_service = rospy.Service(CHANGE_TRACKING_TOPIC, SetBool, self.toggle_tracking)
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        self.start_service = rospy.Service(START_TOPIC, SetBool, self.toggle_start_counting)
        self.end_service = rospy.Service(END_TOPIC, PersonCount, self.count)
        self.detection_pub = rospy.Publisher(DETECTION_TOPIC, Point, queue_size=1)
        
        self.people_poses = []
        self.count_frame = []
        self.track_frame = []

        self.image = None
        self.track = False
        self.start = False
        self.end = False

        def load_models():
            pbar = tqdm.tqdm(total=5, desc="Loading models")

            # Yolo 
            self.model = YOLO("yolov8n.pt")
            pbar.update(1)

            # Media pipe 
            self.pose_model = mp.solutions.pose.Pose(min_detection_confidence=0.8) 
            pbar.update(1)

            # ReID
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

        load_models()

        count_thread = threading.Thread(target=self.run_count)
        track_thread = threading.Thread(target=self.run_track)

        count_thread.start()
        track_thread.start()
        # count_process = multiprocessing.Process(target=self.run_count)
        # track_process = multiprocessing.Process(target=self.run_track)

        # count_process.start()
        # track_process.start()

        try:
            rate = rospy.Rate(60)
            while not rospy.is_shutdown():
                if len(self.count_frame) != 0:
                    cv2.imshow("Detections", self.count_frame)
                    cv2.waitKey(1)
                    # self.image_publisher.publish(self.bridge.cv2_to_imgmsg(self.detections_frame, "bgr8"))

                if len(self.track_frame) != 0:
                    print("track frame showing?")
                    cv2.imshow("Tracking", self.track_frame)
                    cv2.waitKey(1)
                    # self.image_publisher.publish(self.bridge.cv2_to_imgmsg(self.track_frame, "bgr8"))
                    
                rate.sleep()
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()
    
    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def toggle_tracking(self, req):
        self.track = req.data
        return True, "Tracking is now " + ("enabled" if self.track else "disabled")

    def toggle_start_counting(self, req):
        self.start = req.data
        return True, "Counting started"

    # def track_callback(self, data):
    #     self.track = data

    def getCenter(self, box):
        if ARGS["FLIP_IMAGE"]:
            x1 = 1 - int(box[2])
            y1 = 1 - int(box[3])
            x2 = 1 - int(box[0])
            y2 = 1 - int(box[1])

        else:
            x1, y1, x2, y2 = [int(i) for i in box]

        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)

        return x, y
    
    def count(self, req):
        print("People detected", len(self.people_poses))

        people_sitting = 0
        people_standing = 0
        # people_pointing = 0
        pointing_left = 0
        pointing_right = 0
        right_hand = 0
        left_hand = 0
        waving = 0
        shirt_color = ""
        # people_raising_hand = 0

        for (person_poses) in self.people_poses:
            for pose in person_poses:
                if pose == "Sitting":
                    people_sitting += 1
                
                if pose == "Standing":
                    people_standing += 1

                if pose == "Pointing right":
                    pointing_right += 1
                
                if pose == "Pointing left":
                    pointing_left += 1
                
                if pose == "Raising right hand":
                    right_hand += 1
                
                if pose == "Raising left hand":
                    left_hand += 1

                if pose == "Waving":
                    waving += 1

                if pose == "Shirt color":
                    shirt_color += f"{pose},"

                
        self.people_poses = []
        self.start = False
        self.end = False

        if req.data == "Sitting":
            result = str(people_sitting)

        elif req.data == "Standing":
            return str(people_standing)

        elif req.data == "Pointing Left":
            return str(pointing_left)
        
        elif req.data == "Pointing Right":
            return str(pointing_right)
        
        elif req.data == "Raising Right Hand":
            return str(right_hand)
        
        elif req.data == "Raising Left Hand":
            return str(left_hand)
        
        elif req.data == "Waving":
            return str(waving)
        
        elif req.data == "Shirt Color":
            return shirt_color
        
        elif req.data == "Raising hands":
            return str(right_hand + left_hand)
        else:
            return "Request not found"
        

    def run_count(self):       

        # Initialize lists to store information about people
        people_tags = []
        people_ids = []
        people_features = []
        self.people_poses = []
        prev_ids = []

        print('Running count')

        while rospy.is_shutdown() == False :

            if self.start == False:
                people_tags = []
                people_ids = []
                people_features = []
                self.people_poses = []
                prev_ids = []

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


                    print(track_ids)
                    print(people_tags)
                    print(self.people_poses)
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
                    # cv2.imshow("YOLOv8 Tracking", frame)
                    self.count_frame = frame

                    # Break the loop if 'q' is pressed
                    # if cv2.waitKey(1) & 0xFF == ord("q"):
                    #     break

        # Release the video capture object and close the display window
        # cv2.destroyAllWindows()

    def run_track(self):

        # Initialize lists to store information about people
        people_tags = []
        people_ids = []
        people_features = []
        prev_ids = []
        track_person = ""

        # print('Running Track')
        rospy.loginfo("Running Track")

        while rospy.is_shutdown() == False :
            if self.track is False:
                people_tags = []
                people_ids = []
                people_features = []
                prev_ids = []
                track_person = ""
                
            
            else:
                print("Tracking")
                if self.image is not None:
                    rospy.loginfo("Tracking")
                    # print("Tracking ")

                    # Get the frame from the camera
                    frame = self.image
                    if ARGS["FLIP_IMAGE"]:
                        frame = imutils.rotate(frame, 180)

                    width = frame.shape[1]
                    
                    # Get the results from the YOLOv8 model
                    results = self.model.track(frame, persist=True, tracker='bytetrack.yaml', classes=0, verbose=False) #could use botsort.yaml
                    
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
                        if track_id not in prev_ids:

                            # Get bbox
                            x1, y1, x2, y2 = [int(i) for i in box]
                               

                            # Crop the image 
                            cropped_image = frame[y1:y2, x1:x2]
                            # cv2.imshow('crpd',cropped_image)
                            pil_image = PILImage.fromarray(cropped_image)
                            person = check_visibility(self.pose_model,cropped_image)
                            # cv2.imshow('crpd',cropped_image)

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
                                print("New person detected")
                                people_ids.append(track_id)
                                people_tags.append(f"Person {len(people_ids)}")
                                people_features.append(new_feature)
                            
                    print(track_ids)
                    print(people_tags)
                    print(people_ids)
                    prev_ids = []
                    prev_features = []
                    
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
                            cx, cy = self.getCenter(box)
 
                        
                        if w*h > max_area:
                            max_area = w*h
                            index = i
                            # print(area_id)

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
                    # cv2.imshow("Person Tracking", frame)
                    self.track_frame = frame

                    # Break the loop if 'q' is pressed
                    # if cv2.waitKey(1) & 0xFF == ord("q"):
                    #     break
            # else:
            #     # End of video
            #     break

        # Release the video capture object and close the display window
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    try: 
        for key in ARGS:
            ARGS[key] = rospy.get_param('~' + key, ARGS[key])

        PersonTracking()
    except rospy.ROSInterruptException:
        pass