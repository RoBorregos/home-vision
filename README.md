# robocup-home-vision
Computer Vision resources and packages for the Robocup@Home Competition.

# Tasks
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


**Launch Nodes**

``` bash
roslaunch vision receptionist.launch
```

