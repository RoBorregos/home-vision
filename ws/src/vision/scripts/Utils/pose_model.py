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

class Direction(Enum):
    RIGHT = "Pointing right"
    LEFT = "Pointing left"
    NOT_POINTING = "not_pointing"


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
    return degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
    


def classify_pose(poseModel, image, print_angles=False, general=True):
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
        leg_angle_left = getAngle(hip_left, knee_left, ankle_left)

        # Right leg points
        hip_right = landmarks[24]
        knee_right = landmarks[26]  
        ankle_right = landmarks[28]
        leg_angle_right = getAngle(hip_right, knee_right, ankle_right)


        # Left arm points
        shoulder_left = landmarks[11]
        elbow_left = landmarks[13]
        wrist_left = landmarks[15]
        index_left = landmarks[19]
        arm_angle_left = getAngle(shoulder_left, elbow_left, wrist_left)

        # Right arm points
        shoulder_right = landmarks[12]
        elbow_right = landmarks[14]
        wrist_right = landmarks[16]
        index_right = landmarks[20]
        arm_angle_right = getAngle(shoulder_right, elbow_right, wrist_right)


        # Get shirt color
        print("color")
        color = get_shirt_color(image, shoulder_right, shoulder_left, hip_right, hip_left)
        poses.append(f"shirt color: {color}")

        m = 0.1

        right_out = index_right.x*w < shoulder_right.x*w - m*w 
        left_out = index_left.x*w > shoulder_left.x*w + m*w
        

        if right_out and left_out :
            data = Direction.NOT_POINTING
        elif right_out:
            data = Direction.RIGHT
        elif left_out:
            data = Direction.LEFT
        else:
            data = Direction.NOT_POINTING


        # Determine if standing or sitting
        if (leg_angle_left and leg_angle_right) > 120:
            poses.append("Standing")

        else:
            poses.append("Sitting")


        if general:
            if shoulder_right.y - wrist_right.y > RAISING_HAND_THRESHOLD or shoulder_left.y - wrist_left.y > RAISING_HAND_THRESHOLD:
                poses.append("Raising hand/s")

            # if data != Direction.NOT_POINTING:
            #     poses.append("Pointing")

        else:
            # Determine if pointing to a direction or raising hand 
            if data != Direction.NOT_POINTING:
                poses.append(data.value)

            if shoulder_right.y - wrist_right.y > RAISING_HAND_THRESHOLD:
                poses.append("Raising right hand")

            if shoulder_left.y - wrist_left.y > RAISING_HAND_THRESHOLD:
                poses.append("Raising left hand")
            
            

        # Print angles
        if print_angles:
            print("Left leg angle: ", leg_angle_left)
            print("Right leg angle: ", leg_angle_right)
            print("Left arm angle: ", arm_angle_left)
            print("Right arm angle: ", arm_angle_right)


        # Draw results
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        arr = np.array(poses)
        txt = np.array2string(arr)
        cv2.putText(annotated_image, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_image, data.value, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

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
