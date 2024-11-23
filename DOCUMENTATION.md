# ROS Nodes Documentation

## FaceRecognition.py

**Purpose**: Uses the `face_recognition` library to detect and recognize faces in images. It subscribes to the camera topic and publishes the detected person's name and coordinates.

**Input Topics**:
- `/zed2/zed_node/rgb/image_rect_color` (sensor_msgs/Image): The input image from the camera.

**Output Topics**:
- `/vision/person_detected_name` (std_msgs/String): The name of the detected person.
- `/vision/person_list` (frida_vision_interfaces/PersonList): The list of detected persons.
- `/vision/target` (geometry_msgs/Point): The coordinates of the detected person.
- `/vision/img_face_recognition` (sensor_msgs/Image): The annotated image with face recognition results.

**Services**:
- `/vision/new_name` (frida_vision_interfaces/NewHost): Service to add a new person to the face recognition system.

**Parameters**:
- `MAX_DEGREE` (int): The maximum degree for face tracking.
- `TRACK_THRESHOLD` (int): The threshold for tracking faces.
- `AREA_THRESHOLD` (int): The threshold for face area.

## ReceptionistCommands.py

**Purpose**: Uses YOLO models to detect people and objects in images. It provides services to check for the presence of people and find available seats.

**Input Topics**:
- `/zed2/zed_node/rgb/image_rect_color` (sensor_msgs/Image): The input image from the camera.

**Output Topics**:
- `/vision/img_person_detection` (sensor_msgs/Image): The annotated image with person detection results.

**Services**:
- `/vision/check_person` (std_srvs/SetBool): Service to check if a person is detected.
- `/vision/find_seat` (frida_vision_interfaces/FindSeat): Service to find an available seat.

**Parameters**:
- `MODEL_LOCATION` (str): The path to the YOLO model.
- `PERCENTAGE` (float): The percentage of the image to consider for person detection.
- `MAX_DEGREE` (int): The maximum degree for seat finding.

## PersonTracking.py

**Purpose**: Uses YOLO and ReID models to track people in images. It publishes the coordinates of the tracked person.

**Input Topics**:
- `/zed2/zed_node/rgb/image_rect_color` (sensor_msgs/Image): The input image from the camera.

**Output Topics**:
- `/vision/person_detection` (geometry_msgs/Point): The coordinates of the tracked person.
- `/vision/img_tracking` (sensor_msgs/Image): The annotated image with person tracking results.

**Services**:
- `/vision/change_person_tracker_state` (std_srvs/SetBool): Service to enable or disable person tracking.

**Parameters**:
- `FLIP_IMAGE` (bool): Whether to flip the input image.

## bagDetector.py

**Purpose**: Uses YOLO models to detect bags in images. It publishes the detected bags and their 3D coordinates.

**Input Topics**:
- `/zed2/zed_node/rgb/image_rect_color` (sensor_msgs/Image): The input image from the camera.
- `/camera/depth/image_raw` (sensor_msgs/Image): The input depth image from the camera.
- `/camera/depth/camera_info` (sensor_msgs/CameraInfo): The camera info for the depth image.

**Output Topics**:
- `detections/bag` (bag_detector/objectDetectionArray): The detected bags.
- `detections_image/bag` (sensor_msgs/Image): The annotated image with bag detection results.
- `detections_3d/bag` (visualization_msgs/MarkerArray): The 3D coordinates of the detected bags.

**Parameters**:
- `SOURCE` (str): The source of the input image.
- `ROS_INPUT` (bool): Whether to use ROS input.
- `USE_ACTIVE_FLAG` (bool): Whether to use the active flag.
- `DEPTH_ACTIVE` (bool): Whether to use the depth image.
- `DEPTH_INPUT` (str): The input depth image topic.
- `CAMERA_INFO` (str): The camera info topic.
- `MIN_SCORE_THRESH` (float): The minimum score threshold for bag detection.
- `VERBOSE` (bool): Whether to enable verbose output.
- `CAMERA_FRAME` (str): The camera frame.
- `YOLO_BAG_MODEL_PATH` (str): The path to the YOLO bag model.
- `FLIP_IMAGE` (bool): Whether to flip the input image.

## detectPointingObject.py

**Purpose**: Uses Mediapipe to detect the pointing direction of a user's hand. It provides an action server to detect the pointed object.

**Input Topics**:
- `/zed2/zed_node/rgb/image_rect_color` (sensor_msgs/Image): The input image from the camera.
- `detections/bag` (bag_detector/objectDetectionArray): The detected bags.

**Output Topics**:
- `pointing_direction` (sensor_msgs/Image): The annotated image with pointing direction results.
- `pointed_object_marker` (visualization_msgs/Marker): The marker for the pointed object.

**Services**:
- `detectPointingObject` (frida_vision_interfaces/DetectPointingObjectAction): Action server to detect the pointed object.

**Parameters**:
- `USE_RIGHT_HAND` (bool): Whether to use the right hand for pointing detection.
- `USE_LEFT_HAND` (bool): Whether to use the left hand for pointing detection.
- `INFERENCE_TIMEOUT` (int): The timeout for inference.
- `SOURCE` (str): The source of the input image.
- `ROS_INPUT` (bool): Whether to use ROS input.
- `USE_ACTIVE_FLAG` (bool): Whether to use the active flag.
- `DEPTH_ACTIVE` (bool): Whether to use the depth image.
- `DEPTH_INPUT` (str): The input depth image topic.
- `CAMERA_INFO` (str): The camera info topic.
- `MIN_SCORE_THRESH` (float): The minimum score threshold for pointing detection.
- `VERBOSE` (bool): Whether to enable verbose output.
- `CAMERA_FRAME` (str): The camera frame.
- `YOLO_BAG_MODEL_PATH` (str): The path to the YOLO bag model.
- `FLIP_IMAGE` (bool): Whether to flip the input image.
