#!/usr/bin/env python3
import rospy
# from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
# import moveit_commander
from sensor_msgs.msg import JointState
import time
# import tf
# from deepface import DeepFace


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

CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"
NEW_NAME_TOPIC = "/new_name"
PERSON_NAME_TOPIC = "/person_detected_name" 
PERSON_LIST_TOPIC = "/person_list"
TARGET_TOPIC = "/target"
NEW_HOST = "/new_host"


TRACK_THRESHOLD = 50
AREA_THRESHOLD = 1200

# Paths
json_path = str(pathlib.Path(__file__).parent) + "/Utils/known_people/identities.json"
# json_path = "src/vision/scripts"
folder = str(pathlib.Path(__file__).parent) + "/Utils/known_people"
# folder = "src/vision/scripts/known_people"




class FaceRecognition():

    def __init__(self):
        rospy.init_node('face_recognition')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        self.new_name_service = rospy.Service(NEW_NAME_TOPIC, NewHost, self.new_name_callback)
        self.name_pub = rospy.Publisher(PERSON_NAME_TOPIC, String, queue_size=1)
        self.move_pub = rospy.Publisher(TARGET_TOPIC, Point, queue_size=1)
        self.person_list_pub = rospy.Publisher(PERSON_LIST_TOPIC, PersonList, queue_size=1)
        self.new_name = ""
        self.image = None

        # Progress bar
        pbar = tqdm.tqdm(total=2)

        # Load images
        random = face_recognition.load_image_file(f"{folder}/random.png")

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
        print("Face recognition ready")

        self.run()
        

    def new_name_callback(self, req):
        self.new_name = req.name
        return True

    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def new_host_callback(self, data):
        self.new_host = data

    # Make encodings of known people images
    def process_imgs(self):
        print("Processing images")
        for filename in os.listdir(folder):
            if filename == ".DS_Store" or filename == "identities.json":
                continue
            
            self.process_img(filename)

    # Function to clear previous results
    def clear(self):
        for filename in os.listdir(folder):
            if filename == ".DS_Store" or filename == "identities.json" or filename == "random.png" or filename == "Adan.png":
                continue

            self.process_img(filename)

            file_path = os.path.join(folder, filename)
            os.remove(file_path)

        # load json
        

        # f = open(json_path)

        data = {
            "random": {"age": 21, "gender": "female", "race": "race"}
        }

        with open(json_path, 'w') as outfile:
            json.dump(data, outfile)

        print("Cleared")




    def process_img(self, filename):
        img = face_recognition.load_image_file(f"{folder}/{filename}")
        cur_encodings = face_recognition.face_encodings(img)

        if len(cur_encodings) == 0:
            print('no encodings found')
            return
        
        if len(cur_encodings) > 0:
            cur_encodings = cur_encodings[0]

        self.people_encodings.append(cur_encodings)
        self.people_names.append(filename[:-4])
        self.people.append([cur_encodings, filename[:-4]])

        print(f"{folder}/{filename}")

    def upadate_json(self, face_id, image):

        # load json
        print("updating json")

        f = open(json_path)
        data = json.load(f)
        print(face_id)
        if face_id not in data:
            try:
                # features_list = DeepFace.analyze(image, enforce_detection=True)
                # print(f"features list size is {len(features_list)}")
                # features = features_list[0]
                # age = features.get('age')
                # gender = features.get('dominant_gender')
                # race = features.get('dominant_race')
                # emotions = features.get('dominant_emotion')


                data[face_id] = {
                    "age": 12,
                    "gender": "gender",
                    "race": "race"
                }

                with open(json_path, 'w') as outfile:
                    json.dump(data, outfile)

                print("updated json")
            except:
                print("error getting attributes")
                # faces_tracked.remove(face_id)

        

    def run(self):
            
        self.clear()

        prev_faces = [] 
        curr_faces = []
        print("Running face recognition")
        while rospy.is_shutdown() == False :

            # img_arr = img_list()

            if self.image is not None:
                # print('img')
                frame = self.image
               
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                center = [frame.shape[1]/2, frame.shape[0]/2]
                # print(center[0], ", ", center[1])
                
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(small_frame)
                # print("detected: ", len(face_locations))

                largest_area = 0
                largest_face_params = None
                largest_face_name = ""

                annotated_frame = frame.copy()
                
                curr_faces = []
                face_list = PersonList()
                detected = False
                max_degree = 30

                face_encodings = None
                for i, location in enumerate(face_locations):
                    # print('detected: ', len(face_locations))
                # for face_encoding, location in zip(face_encodings, face_locations):

                    # print("l____",location[0])
                    # print("detected: ", len(detected_faces))

                    # Center of current face
                    centerx = (location[3] + (location[1] - location[3])/2)*2
                    centery = (location[0] + (location[2] - location[0])/2)*2
                    # print(location[0], 'bottom: ', location[2])

                    top = location[0]*2
                    right = location[1]*2
                    bottom = location[2]*2
                    left = location[3]*2

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

                        # print("detected: ", prev_face["x"], "center: ", centerx, "diff: ", abs(prev_face["x"] - centerx))
                        # print("detected: ", prev_face["y"], "center: ", centery, "diff: ", abs(prev_face["y"] - centery))
                        # print('x', centerx, '   y', centery)

                        # If the face is within the tracking threshold
                        if (abs(prev_face["x"] - centerx) < TRACK_THRESHOLD) and (abs(prev_face["y"] - centery) < TRACK_THRESHOLD):
                            name = prev_face["name"]
                            # print("Tracking: ", name)
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
                            # print("Matched")
                            name = self.people_names[best_match_index]
            

                    xc = left + (right - left)/2
                    yc = top + (bottom - top)/2
                    area = (right-left)*(bottom-top)
                    # print(xc)

                    curr_faces.append({"x": xc, "y": yc, "name": name})

                    curr_person = Person()
                    curr_person.name = name
                    curr_person.x = int( (xc-center[0]) * max_degree/center[0] )
                    curr_person.y = int( (center[1]-yc) * max_degree/center[1] )

                    face_list.list.append(curr_person)

                    detected = True

                    if (area > largest_area):
                        largest_area = area
                        largest_face_params = [left, top, right, bottom]
                        largest_face_name = name

                    # Show results ______________________________________________
                    
                    if flag:
                        cv2.rectangle(annotated_frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
                        cv2.rectangle(annotated_frame, (left, top), (right, bottom), (255, 0, 0), 2)
                    else:
                        cv2.rectangle(annotated_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX

                    cv2.putText(annotated_frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    # cv2.circle(annotated_frame, (int(xc), int(yc)), 5, (0, 255, 0), -1)

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
                        # if area < AREA_THRESHOLD:
                        #     print("Person too small")

                        # else:
                        largest_face_name = self.new_name
                        crop = frame[top:bottom, left:right]

                        img_name = f"{largest_face_name}.png"
                        save_path = f"{folder}/{img_name}"
                        cv2.imwrite(save_path, crop)
                        self.process_img(img_name)
                        self.upadate_json(img_name, crop)

                        # Update prev recognitions for tracker
                        for i, face in enumerate(curr_faces):
                            if face["x"] == xc and face["y"] == yc:
                                index = i
                                # print("found")

                            cv2.circle(annotated_frame, (int(face["x"]), int(face["y"])), 5, (0, 255, 0), -1)

                        curr_faces[index]["name"] = self.new_name
                        face_list.list[index].name = self.new_name
                        print(f"{self.new_name} face saved")
                        self.new_name = ""

                prev_faces = curr_faces
                # print(len(curr_faces))

                # Publish coordinates to track face
                if xc != 0:
                    difx = xc - center[0] 
                else:
                    difx = 0
                
                if yc != 0:
                    dify = center[1] - yc
                else:
                    dify = 0
                    
                max_degree = 30

                if detected:
                    # Calculate the movement for the camera to track the face
                    move_x = difx*max_degree/center[0]
                    move_y = dify*max_degree/center[1]

                    move = Point()
                    
                    move.x = int(move_x)
                    move.y = int(move_y)

                    self.move_pub.publish(move)

                    person_seen = String()
                    person_seen.data = largest_face_name

                    self.name_pub.publish(person_seen)
                    self.person_list_pub.publish(face_list)

                cv2.imshow("Face detection", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    prev_faces = []
                    break
                
            
                  
def main():
    # for key in ARGS:
    #     ARGS[key] = rospy.get_param('~' + key, ARGS[key])
    FaceRecognition()


if __name__ == '__main__':
    main()