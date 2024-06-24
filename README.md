# robocup-home-vision
Computer Vision resources and packages for the Robocup@Home Competition.

# Docker
To launch vision nodes, it is recommended to use the dockerfiles according to the computer's resources (CUDA):

Without cuda:
``` bash
make vision.build
make vision.create
make vision.shell
```

With cuda:
``` bash
make vision.build.cuda
make vision.create.cuda
make vision.shell
```

# Launch files
## Receptionist
For the receptionist task, it was required to recognize people, identify empty seats from either chairs or couches and detect if a person is standing in front of a robot. Hence, two nodes where developed. One to perform person or object detection procedures and one for the face-recognition analysis. These functionalities where accessed through three services:

**Person Detection**
- Identifies faces and publishes points for arm to follow the largest face.
- Recieves name from service to save a face

Service topic: `/vision/check_person`
  
**Face Recognition**
- Identifies faces and publishes points for arm to follow the largest face.
- Recieves name from service to save a face

Service topic: `/vision/new_name`

**Seat Finding**
- Identifies faces and publishes points for arm to follow the largest face.
- Recieves name from service to save a face

Service topic: `/vision/find_seat`


**Launch File**

``` bash
roslaunch vision receptionist.launch
```

## Tracker
For several tasks, it is necessary to identify a person and track their position. Hence a node was developed using YOLO and a REID model.

Service topic: `/vision/change_person_tracker_state`

**Launch File**

``` bash
roslaunch vision tracking.launch FLIP_IMAGE:=false
```

## Person Commands (GPSR)
For the GPSR task, there are several commands that require computer vision algorithms. Some involve identifying a person who meets a specific pose or shirt color, and some involving counting people who meet this criteria. Therefore, the person command node has several services for this commands. It uses YOLOv8 for person detection, Mediapipe for pose detection and a custom model for shirt color detection.

**Person counting**

Service topic (bool): `/vision/start_counting`
Service topic (string): `/vision/end_counting`

**Person finding**

Service topic (string): `/vision/find_pose`

## Shelf
**Shelf detection**
