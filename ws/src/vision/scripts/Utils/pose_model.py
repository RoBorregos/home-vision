import mediapipe as mp
import numpy as np
import cv2
from math import acos, degrees
import os
from enum import Enum
from .shirt_color import get_shirt_color
'''
Script to classify the pose of a person in an image:
Standing, Sitting, Raising right hand, Raising left hand, Pointing left, Pointing right
'''

SITTING_THRESHOLD = 90
RAISING_HAND_THRESHOLD = 0.01
POINTING_THRESHOLD = 165
WAVING_THRESHOLD = 160

class Direction(Enum):
    NOT_POINTING = 0
    RIGHT = 1
    LEFT = 2


def check_visibility(poseModel, image):
    pose = poseModel
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image
    results = pose.process(image)
    # Check if the pose landmarks are detected
    if results.pose_landmarks is not None:
        # Get the x and y coordinates of the chest and face landmarks
        chest_x = results.pose_landmarks.landmark[11].x
        chest_y = results.pose_landmarks.landmark[11].y
        chest_visibility = results.pose_landmarks.landmark[11].visibility

        mp_drawing = mp.solutions.drawing_utils
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        # Convert the image back to BGR
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        # Display the annotated image
        # cv2.imshow("Annotated Image", annotated_image)

        if (chest_x < 0 or chest_x > 1 or chest_y < 0 or chest_y > 1) and chest_visibility < 0.95:
            # print("Chest not visible")
            return False
        else:
            # print("Chest visible")
            return True
            
def getCenterPerson(poseModel, image):
    # Process the image
    results = poseModel.process(image)

    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark
        shoulder_right = landmarks[12]
        shoulder_left = landmarks[11]

        x_center = (shoulder_right.x + shoulder_left.x) / 2
        y_center = (shoulder_right.y + shoulder_left.y) / 2
        
        cv2.circle(image, (int(x_center * image.shape[1]), int(y_center * image.shape[0])), 5, (0, 0, 255), -1)
        cv2.imshow("Annotated Image", image)
        
        return x_center*image.shape[1], y_center*image.shape[0]
    
    return None, None



def getAngle(point_close, point_mid, point_far):
    # Convert the points to numpy arrays
    p1 = np.array([point_close.x, point_close.y])
    p2 = np.array([point_mid.x, point_mid.y])
    p3 = np.array([point_far.x, point_far.y])

    # Euclidean distances
    l1 = np.linalg.norm(p2 - p3)    #lados de triangulo
    l2 = np.linalg.norm(p1 - p3)
    l3 = np.linalg.norm(p1 - p2)

    # Cálculo de ángulo pierna izquierda
    return abs(degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3))))
    


def classify_pose(poseModel, image, print_angles=False, general=False):
    poses = []
    # Convert the image to RGB
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Process the image
    results = poseModel.process(image)

    # Get the landmarks
    landmarks = results.pose_landmarks

    # Get the height and width of the image
    h = image.shape[0]
    w = image.shape[1]

    if landmarks is not None:
        print("f")
        landmarks = landmarks.landmark
        # Left leg points
        hip_left = landmarks[23]
        knee_left = landmarks[25]
        ankle_left = landmarks[27]

        # Right leg points
        hip_right = landmarks[24]
        knee_right = landmarks[26]  
        ankle_right = landmarks[28]


        # Left arm points
        shoulder_left = landmarks[11]
        elbow_left = landmarks[13]
        wrist_left = landmarks[15]
        index_left = landmarks[19]
        

        # Right arm points
        shoulder_right = landmarks[12]
        elbow_right = landmarks[14]
        wrist_right = landmarks[16]
        index_right = landmarks[20]

        # Get angles
        leg_angle_left = getAngle(hip_left, knee_left, ankle_left)
        leg_angle_right = getAngle(hip_right, knee_right, ankle_right)
        elbow_angle_left = getAngle(shoulder_left, elbow_left, wrist_left)
        shoulder_angle_left = getAngle(shoulder_right, shoulder_left, elbow_left)
        elbow_angle_right = getAngle(shoulder_right, elbow_right, wrist_right)
        shoulder_angle_right = getAngle(shoulder_left, shoulder_right, elbow_right)


        # Get shirt color
        color = get_shirt_color(image, shoulder_right, shoulder_left, hip_right, hip_left)
        poses.append(color)


        # Determine if standing or sitting
        if (leg_angle_left and leg_angle_right) > 120:
            poses.append("Standing")

        else:
            poses.append("Sitting")


        if general:
            if shoulder_right.y - wrist_right.y > RAISING_HAND_THRESHOLD or shoulder_left.y - wrist_left.y > RAISING_HAND_THRESHOLD:
                poses.append("Raising hands")

            # if data != Direction.NOT_POINTING:
            #     poses.append("Pointing")

        else:
            # Determine if pointing to a direction or raising hand 
            if elbow_angle_right > POINTING_THRESHOLD and shoulder_angle_right > POINTING_THRESHOLD:
                poses.append("Pointing right")
            
            if elbow_angle_left > POINTING_THRESHOLD and shoulder_angle_left > POINTING_THRESHOLD:
                poses.append("Pointing left")

            if shoulder_right.y - wrist_right.y > RAISING_HAND_THRESHOLD:
                poses.append("Raising right hand")

                if elbow_angle_right < WAVING_THRESHOLD:
                    poses.append("Waving")

            if shoulder_left.y - wrist_left.y > RAISING_HAND_THRESHOLD:
                poses.append("Raising left hand")

                if elbow_angle_left < WAVING_THRESHOLD and "Waving" not in poses:
                    poses.append("Waving")


        # Draw results
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        arr = np.array(poses)
        # txt = np.array2string(arr)
        # cv2.putText(annotated_image, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.putText(annotated_image, data.value, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        # Show image
        # cv2.imshow("annotated_image.jpg", annotated_image)
        # cv2.waitKey(0)
        
    print(poses)
    return poses
        

# # Main
# pose_model = mp.solutions.pose.Pose(min_detection_confidence=0.8) 

# folder = "./images"
# for filename in (os.listdir(folder)):

#     path = os.path.join(folder, filename)
#     img = cv2.imread(path)
#     poses = classify_pose(pose_model, img)
#     print(filename, " : " , poses)
