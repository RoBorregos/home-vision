#!/usr/bin/env python3
import rospy
import time
from PIL import Image as PILImage
import cv2
from ultralytics import YOLO

from Utils.reid_model import load_network, compare_images, extract_feature_from_img, get_structure
from Utils.pose_model import check_visibility, classify_pose
import torch.nn as nn
import torch
import tqdm
import mediapipe as mp

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from vision.srv import PersonCount
from vision.msg import people_count

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
THRESHOLD = 25

class PersonCounting():

    def __init__(self):
        rospy.init_node('person_counting')
        self.bridge = CvBridge()
        self.start_service = rospy.Service(START_TOPIC, SetBool, self.toggle_start)
        self.end_service = rospy.Service(END_TOPIC, PersonCount, self.count)
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        self.start_sub = rospy.Subscriber(START_TOPIC, Bool, self.start_callback)
        self.end_sub = rospy.Subscriber(END_TOPIC, Bool, self.end_callback)
        self.count_pub = rospy.Publisher(RESULTS_TOPIC, people_count, queue_size=1)
        self.image = None
        self.start = False
        self.end = False
    
    def toggle_start(self, req):
        self.start = req.data
        return True, "Counting started"
    
    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def start_callback(self, data):
        self.start = data

    def end_callback(self, data):
        self.end = data
    
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

    
    def run(self):
        pbar = tqdm.tqdm(total=5, desc="Loading models")

        # Load the YOLOv8 model
        model = YOLO('yolov8n.pt')
        pbar.update(1)

        # Load media pipe model
        pose_model = mp.solutions.pose.Pose(min_detection_confidence=0.8) 
        pbar.update(1)

        # Load the ReID model
        structure = get_structure()
        pbar.update(1)
        model_reid = load_network(structure)
        pbar.update(1)
        model_reid.classifier.classifier = nn.Sequential()
        pbar.update(1)
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model_reid = model_reid.cuda()
        pbar.close()

        # Initialize lists to store information about people
        people_tags = []
        people_ids = []
        people_features = []
        self.people_poses = []
        prev_ids = []

        print('Running')

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
                    results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=0, verbose=False) #could use botsort.yaml
                    
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
                            person = check_visibility(pose_model,cropped_image)

                            # Check if the person is visible enough (chest) and not too close to the edges of the image
                            if not person or x1 <= THRESHOLD or x2 >= width - THRESHOLD:
                                false_detections.append(track_id)
                                continue

                            # Get feature for REID model
                            with torch.no_grad():
                                new_feature = extract_feature_from_img(pil_image, model_reid)
                            flag = False

                            # Get pose
                            pose = classify_pose(pose_model, cropped_image)

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
                    cv2.imshow("YOLOv8 Tracking", frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        # Release the video capture object and close the display window
        cv2.destroyAllWindows()

# Run node
trackNode = PersonCounting()
trackNode.run()
