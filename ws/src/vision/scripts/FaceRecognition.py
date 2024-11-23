#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
import time

import cv2
import numpy as np
import os

import face_recognition
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Point
from frida_vision_interfaces.msg import Person, PersonList
from frida_vision_interfaces.srv import NewHost
import tqdm
import json
import pathlib


'''
If name of person -> assign to person in front and save
Check if there are matches and if so, publish the name
Check the largest face detection and post cx,cy
'''

# Service topics
NEW_NAME_TOPIC = "/vision/new_name"

# Subscriber topics
CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"

# Publisher topics
PERSON_NAME_TOPIC = "/vision/person_detected_name" 
PERSON_LIST_TOPIC = "/vision/person_list"
TARGET_TOPIC = "/vision/target"
IMAGE_VIEW = "/vision/img_face_recognition"

# Constants
MAX_DEGREE = 38
TRACK_THRESHOLD = 50
AREA_THRESHOLD = 1200

# Paths
# json_path = str(pathlib.Path(__file__).parent) + "/Utils/known_people/identities.json"
FOLDER = str(pathlib.Path(__file__).parent) + "/Utils/known_people"

class FaceRecognition():

    def __init__(self):
        rospy.init_node('face_recognition')
        self.bridge = CvBridge()

        self.new_name_service = rospy.Service(NEW_NAME_TOPIC, NewHost, self.new_name_callback)
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        self.name_pub = rospy.Publisher(PERSON_NAME_TOPIC, String, queue_size=1)
        self.move_pub = rospy.Publisher(TARGET_TOPIC, Point, queue_size=1)
        self.person_list_pub = rospy.Publisher(PERSON_LIST_TOPIC, PersonList, queue_size=1)
        self.view_pub = rospy.Publisher(IMAGE_VIEW, Image, queue_size=1)

        self.new_name = ""
        self.image_view = None
        self.image = None

        # Progress bar
        pbar = tqdm.tqdm(total=2)

        # Load images
        random = face_recognition.load_image_file(f"{FOLDER}/random.png")

        # Encodings
        random_encodings = face_recognition.face_encodings(random)[0]
        pbar.update(1)

        # Name people and encodings
        self.people = [
            [random_encodings, "random"]
        ]
        self.people_encodings = [
            random_encodings
        ]
        self.people_names = [
            "random"
        ]
        self.clear()
        rospy.loginfo("Face Recognition Ready")

        self.process_imgs()
        self.run()
        

    def new_name_callback(self, req):
        rospy.loginfo("New face request")
        self.new_name = req.name
        return True

    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    # Make encodings of known people images
    def process_imgs(self):
        print("Processing images")
        for filename in os.listdir(FOLDER):
            if filename == ".DS_Store":
                continue
            
            self.process_img(filename)

    # Function to clear previous results
    def clear(self):
        for filename in os.listdir(FOLDER):
            if filename == ".DS_Store" or filename == "random.png" or filename == "Ale.png":
                continue

            self.process_img(filename)

            file_path = os.path.join(FOLDER, filename)
            os.remove(file_path)

        print("Cleared")
    
    # Process image and add to known people
    def process_img(self, filename):
        img = face_recognition.load_image_file(f"{FOLDER}/{filename}")
        cur_encodings = face_recognition.face_encodings(img)

        if len(cur_encodings) == 0:
            print('no encodings found')
            return
        
        if len(cur_encodings) > 0:
            cur_encodings = cur_encodings[0]

        self.people_encodings.append(cur_encodings)
        self.people_names.append(filename[:-4])
        self.people.append([cur_encodings, filename[:-4]])

        print(f"{FOLDER}/{filename}")

    
    # Main function to run face recognition
    def run(self):
            
        self.clear()
        prev_faces = [] 
        curr_faces = []
        rospy.loginfo("Running Face Recognition")

        while rospy.is_shutdown() == False :

            if self.image is not None:
                # print('img')
                frame = self.image
               
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                center = [frame.shape[1]/2, frame.shape[0]/2]
                
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(small_frame)
                
                largest_area = 0
                largest_face_params = None
                largest_face_name = ""

                annotated_frame = frame.copy()
                curr_faces = []
                face_list = PersonList()
                detected = False
                face_encodings = None
                
                for i, location in enumerate(face_locations):

                    # Center of current face
                    centerx = (location[3] + (location[1] - location[3])/2)*2
                    centery = (location[0] + (location[2] - location[0])/2)*2

                    top, right, bottom, left = [i * 2 for i in location]
                    name = "Unknown"
                    
                    # Extend bbox
                    left = max(left - 50,0)
                    right = right + 50
                    top = max(0,top - 50)
                    bottom = bottom + 50
                    area = (right-left)*(bottom-top)

                    # Tracking
                    flag = False
                    for prev_face in prev_faces:

                        # If the face is within the tracking threshold
                        if (abs(prev_face["x"] - centerx) < TRACK_THRESHOLD) and (abs(prev_face["y"] - centery) < TRACK_THRESHOLD):
                            name = prev_face["name"]
                            flag = True
                            break

                    # If not a tracked face, then it needs to be processed (compare to known faces)
                    if not flag:
                        if face_encodings == None:
                            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
                        
                        face_encoding = face_encodings[i]

                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(face_encoding, self.people_encodings, 0.5)
                        face_distances = face_recognition.face_distance(self.people_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        # If it is known, then the name is updated
                        if matches[best_match_index]:
                            name = self.people_names[best_match_index]
            

                    xc = left + (right - left)/2
                    yc = top + (bottom - top)/2
                    area = (right-left)*(bottom-top)

                    # Add face to list
                    curr_faces.append({"x": xc, "y": yc, "name": name})
                    curr_person = Person()
                    curr_person.name = name
                    curr_person.x = int( (xc-center[0]) * MAX_DEGREE/center[0] )
                    curr_person.y = int( (center[1]-yc) * MAX_DEGREE/center[1] )
                    face_list.list.append(curr_person)

                    detected = True

                    if (area > largest_area):
                        largest_area = area
                        largest_face_params = [left, top, right, bottom]
                        largest_face_name = name

                    # Show results 
                    if flag:
                        cv2.rectangle(annotated_frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
                        cv2.rectangle(annotated_frame, (left, top), (right, bottom), (255, 0, 0), 2)
                    else:
                        cv2.rectangle(annotated_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(annotated_frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                xc = 0
                yc = 0

                # For the largest face detected:
                if (largest_area != 0):
                    # Center of the face
                    left, top, right, bottom = largest_face_params
                    xc = left + (right - left)/2
                    yc = top + (bottom - top)/2
                    name = largest_face_name

                    # If received host name
                    if self.new_name != "":
                        largest_face_name = self.new_name
                        crop = frame[top:bottom, left:right]

                        img_name = f"{largest_face_name}.png"
                        save_path = f"{FOLDER}/{img_name}"
                        cv2.imwrite(save_path, crop)
                        self.process_img(img_name)

                        # Update prev recognitions for tracker
                        for i, face in enumerate(curr_faces):
                            if face["x"] == xc and face["y"] == yc:
                                index = i

                            cv2.circle(annotated_frame, (int(face["x"]), int(face["y"])), 5, (0, 255, 0), -1)

                        curr_faces[index]["name"] = self.new_name
                        face_list.list[index].name = self.new_name
                        rospy.loginfo(f"{self.new_name} face saved")
                        
                        self.new_name = ""

                prev_faces = curr_faces

                # Publish coordinates to track face
                if xc != 0:
                    difx = xc - center[0] 
                else:
                    difx = 0
                
                if yc != 0:
                    dify = center[1] - yc
                else:
                    dify = 0
                    
                

                # Calculate the movement for the camera to track the face
                if detected:
                    move_x = difx*MAX_DEGREE/center[0]
                    move_y = dify*MAX_DEGREE/center[1]

                    move = Point()
                    
                    move.x = int(move_x)
                    move.y = int(move_y)

                    self.move_pub.publish(move)

                    person_seen = String()
                    person_seen.data = largest_face_name

                    self.name_pub.publish(person_seen)
                    self.person_list_pub.publish(face_list)

                cv2.imshow("Face detection", annotated_frame)
                self.image_view = annotated_frame
                self.view_pub.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    prev_faces = []
                    break
                
            
                  
def main():
    # for key in ARGS:
    #     ARGS[key] = rospy.get_param('~' + key, ARGS[key])
    FaceRecognition()


if __name__ == '__main__':
    main()