#!/usr/bin/env python3
import rospy
import time
from PIL import Image as PILImage
import cv2
from ultralytics import YOLO
# from ReID import reid_model

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
from std_srvs.srv import SetBool
import imutils

ARGS = {
    "FLIP_IMAGE": False
}

CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"
TRACKING_TOPIC = "/track_person"
DETECTION_TOPIC = "/person_detection"
CHANGE_TRACKING_TOPIC = "/change_person_tracker_state"
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
        # self.tracker_sub = rospy.Subscriber(TRACKING_TOPIC, Bool, self.track_callback)
        self.detection_pub = rospy.Publisher(DETECTION_TOPIC, Point, queue_size=1)
        self.image = None
        self.track = False

        self.run()
    
    def toggle_tracking(self, req):
        self.track = req.data
        return True, "Tracking is now " + ("enabled" if self.track else "disabled")

    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

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
        prev_ids = []
        track_person = ""

        print('Running')

        while rospy.is_shutdown() == False :
            if self.track is False:
                people_tags = []
                people_ids = []
                people_features = []
                prev_ids = []
                track_person = ""
                
            
            else:
                if self.image is not None:
                    # print("Tracking ")

                    # Get the frame from the camera
                    frame = self.image
                    if ARGS["FLIP_IMAGE"]:
                        frame = imutils.rotate(frame, 180)

                    width = frame.shape[1]
                    
                    # Get the results from the YOLOv8 model
                    results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=0, verbose=False) #could use botsort.yaml
                    
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
                            cv2.imshow('crpd',cropped_image)
                            pil_image = PILImage.fromarray(cropped_image)
                            person = check_visibility(pose_model,cropped_image)
                            # cv2.imshow('crpd',cropped_image)

                            if not person or x1 <= THRESHOLD or x2 >= width - THRESHOLD:
                                false_detections.append(track_id)
                                continue

                            # Get feature
                            with torch.no_grad():
                                new_feature = extract_feature_from_img(pil_image, model_reid)
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
                    cv2.imshow("Person Tracking", frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            # else:
            #     # End of video
            #     break

        # Release the video capture object and close the display window
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try: 
        for key in ARGS:
            ARGS[key] = rospy.get_param('~' + key, ARGS[key])

        PersonTracking()
    except rospy.ROSInterruptException:
        pass