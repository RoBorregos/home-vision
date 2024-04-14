#!/usr/bin/env python3
from ultralytics import YOLO
import cv2
import copy 
import math
from Utils.calculations import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import rospy
import time
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, Point, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from vision.msg import objectDetection, level, shelf
import numpy as np


ARGS= {
    "CAMERA_TOPIC": "/zed2/zed_node/rgb/image_rect_color",
    "ROS_INPUT": False,
    "DEPTH_ACTIVE": True,
    "DEPTH_INPUT": "/zed2/zed_node/depth/depth_registered",
    "CAMERA_INFO": "/zed2/zed_node/depth/camera_info",
    "MIN_SCORE_THRESH": 0.5,
    "VERBOSE": True,
    "CAMERA_FRAME": "zed2_left_camera_optical_frame",
    # "YOLO_MODEL_PATH": str(pathlib.Path(__file__).parent) + "/../models/yolov5s.pt",
    "FLIP_IMAGE": False,
    "MIN_CLUSTERS": 2,
    "MAX_CLUSTERS": 6,
    "RESULTS_TOPIC": "/shelf_detection",
    "OUTPUT_IMAGE": "/shelf_detection_image",
    "OUTPUT_3D": "/shelf_detection_3d"
}



ACTIVE_SERVICE_TOPIC = "/shelf_detection_active"

colors = colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0),   # Olive
    (255,255,255)   # White
]


class ShelfDetection():
    def __init__(self):
        rospy.init_node('shelf_detection')
        self.image_sub = rospy.Subscriber(ARGS["CAMERA_TOPIC"], Image, self.image_callback)
        self.depth_sub = rospy.Subscriber(ARGS["DEPTH_INPUT"], Image, self.depthImageRosCallback)
        self.info_sub = rospy.Subscriber(ARGS["CAMERA_INFO"], CameraInfo, self.infoImageRosCallback)
        self.results_pub = rospy.Publisher(ARGS["RESULTS_TOPIC"], shelf, queue_size=1)
        self.results_3d_pub = rospy.Publisher(ARGS["OUTPUT_3D"], MarkerArray, queue_size=1)
        self.image_pub = rospy.Publisher(ARGS["OUTPUT_IMAGE"], Image, queue_size=1)

        self.bridge = CvBridge()
        self.imageInfo = CameraInfo()
        self.model = YOLO('yolov8n.pt')

        self.detections_frame = []
        self.depth_image = []
        self.detections_frame = []

        # Visualization
        try:
            rate = rospy.Rate(60)

            while not rospy.is_shutdown():
                if ARGS["VERBOSE"] and len(self.detections_frame) != 0:
                # if VERBOSE and self.detections_frame != None:
                    cv2.imshow("Detections", self.detections_frame)
                    cv2.waitKey(1)

                if len(self.detections_frame) != 0:
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.detections_frame, "bgr8"))
                    
                rate.sleep()
        except KeyboardInterrupt:
            pass

        cv2.destroyAllWindows()

    # Callbacks
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.run(frame)

    def depthImageRosCallback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)

    def infoImageRosCallback(self, data):
        self.imageInfo = data
        # self.info_sub.unregister()

    # YOLO Functions
    def yolo_run_inference_on_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame)

        output = {
            'detection_boxes': [],  # Normalized ymin, xmin, ymax, xmax
            'detection_classes': [], # ClassID 
            'detection_names': [], # Class Name
            'detection_scores': [] # Confidence
        }
        
        for *xyxy, conf, cls,names in results.pandas().xyxy[0].itertuples(index=False):
            # Normalized [0-1] ymin, xmin, ymax, xmax
            height = frame.shape[1]
            width = frame.shape[0]
            if conf < ARGS["MIN_SCORE_THRESH"]:
                continue
            output['detection_boxes'].append([xyxy[1]/width, xyxy[0]/height, xyxy[3]/width, xyxy[2]/height])
            output['detection_classes'].append(cls)
            output['detection_names'].append(names)
            output['detection_scores'].append(conf)
            # # ClassID
            # found = False
            # count = 1
            # for i in self.category_index.values():
            #     if i == cls:
            #         found = True
            #         break
            #     count += 1
            # if not found:
            #     self.category_index[count] = cls
            # output['detection_classes'].append(count)
            # # Confidence
            # output['detection_scores'].append(conf)
        output['detection_boxes'] = np.array(output['detection_boxes'])
        output['detection_classes'] = np.array(output['detection_classes'])
        output['detection_names'] = np.array(output['detection_names'])
        output['detection_scores'] = np.array(output['detection_scores'])
        return output
    
    def yolov8_run_inference_on_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame)
        output = {
            'detection_boxes': [],  # Normalized ymin, xmin, ymax, xmax
            'detection_classes': [], # ClassID 
            'detection_scores': [], # Confidence
            'detection_names': []
        }

        height = frame.shape[0]
        width = frame.shape[1]

        # print(f"Height: {height} Width: {width}")
        
        boxes = []
        confidences = []
        class_ids = []
        class_names = []

        for out in results:
            for box in out.boxes:
                x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                x1 = x1/width
                x2 = x2/width
                y1 = y1/height
                y2 = y2/height

                class_id = box.cls[0].item()
                prob = round(box.conf[0].item(), 2)
                if prob < ARGS["MIN_SCORE_THRESH"]:
                    continue

                boxes.append([y1, x1, y2, x2])
                confidences.append(float(prob))
                class_ids.append(class_id)
                class_names.append(self.model.names[class_id])
                # print(f"Found {class_id} in {x1} {y1} {x2} {y2}")
                # print("------------------------------")
                # print()

        output['detection_boxes'] = np.array(boxes)
        output['detection_classes'] = np.array(class_ids)
        output['detection_names'] = np.array(class_names)
        output['detection_scores'] = np.array(confidences)
        return output
    
    
    # Handle the detection model input/output.
    def compute_result(self, frame):
        visual_frame = copy.deepcopy(frame)
        # visual_detections = self.yolo_run_inference_on_image(visual_frame)
        visual_detections = self.yolov8_run_inference_on_image(visual_frame)

        detections = copy.deepcopy(visual_detections)

        detections["detection_boxes"] = np.array([[1 - box[2], 1 - box[3], 1 - box[0], 1 - box[1]] for box in visual_detections["detection_boxes"]]) if ARGS["FLIP_IMAGE"] else visual_detections["detection_boxes"]
        
        return detections, visual_detections, visual_frame


    def visualize_detections(self, image, boxes, classes, names, scores, use_normalized_coordinates=True, max_boxes_to_draw=200, min_score_thresh=0.5, agnostic_mode=False):
        """Visualize detections on an input image."""
        
        # Convert image to BGR format (OpenCV uses BGR instead of RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        for i in range(min(boxes.shape[0], max_boxes_to_draw)):
            if scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                if agnostic_mode:
                    color = (0, 0, 0)
                else:
                    color = colors[10]
                ymin, xmin, ymax, xmax = box
                if use_normalized_coordinates:
                    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
                else:
                    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
                (left, right, top, bottom) = (int(left * image.shape[1]), int(right * image.shape[1]),
                                            int(top * image.shape[0]), int(bottom * image.shape[0]))
                cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                # draw label, name and score
                cv2.putText(image, f"{classes[i]}: {names[i]}: {scores[i]}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # draw centroid 
                cv2.circle(image, (int((left + right) / 2), int((top + bottom) / 2)), 5, (0, 0, 255), -1)
        
        # Convert image back to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def cluster_objects(self, detected_objects, frame):
        # Generate example data (y-coordinates of objects)

        y_low = []
        y_mid = []
        indexes = {}

        # Get detection boxes, names and scores
        boxes = detected_objects["detection_boxes"]
        names = detected_objects["detection_names"]
        scores = detected_objects["detection_scores"]
         
        # Make array of y_min and y_low
        for idx, box in enumerate(boxes):
            ymin, xmin, ymax, xmax = box
            y_mid.append((ymin + ymax) / 2)
            y_low.append(ymax)
            indexes[ymin] = idx

        # Check if there are enough objects to cluster
        if len(y_low) < 2:
            print("Not enough objects to cluster")
            return

        # Reshape data into a numpy array (required for KMeans)
        y_low_array = np.array(y_low).reshape(-1, 1)
        optima_clusters = None
        optimal_cluster_labels = None
        max_silhouette_score = -1
        max_clusters = min(ARGS["MAX_CLUSTERS"], len(y_low)-1)

        # Determine the optimal number of clusters
        for n_clusters in range(ARGS["MIN_CLUSTERS"], max_clusters+1):
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            kmeans.fit(y_low_array)
            cluster_labels = kmeans.labels_
            silhouette_avg = silhouette_score(y_low_array, cluster_labels)

            if silhouette_avg > max_silhouette_score:
                max_silhouette_score = silhouette_avg
                optima_clusters = n_clusters
                optimal_cluster_labels = cluster_labels

        # If no optimal number of clusters found, return
        if optima_clusters is None:
            print("No optimal number of clusters found")
            return
        
        # Group objects by cluster
        clusters = {}
        avg_heights = {}
        
        for idx, label in enumerate(optimal_cluster_labels):
            # Make new label if not in clusters
            if label not in clusters:
                clusters[label] = []
                avg_heights[label] = 0

            # Append name, bbox and score
            clusters[label].append({
                "name": names[idx],
                "bbox": boxes[idx],
                "score": scores[idx]
                })
            
            # Add height to cluster for mean calculation
            # avg_heights[label] += y_low[idx] 
            avg_heights[label] += y_mid[idx] 
            

        # Print the clusters
        print(f"Optimal number of clusters: {optima_clusters}")
        annotated_frame = frame

        i = 0
        shelf_levels = shelf()
        shelf_levels.levels = []

        # Make msg for each cluster
        for cluster_label, cluster_items in clusters.items():
            curr_level = level()
            curr_level.label = f"Level {cluster_label}"
            curr_level.mid_height = avg_heights[cluster_label]/len(clusters[cluster_label])
            curr_level.detections = []
            
            print(f"Cluster {cluster_label} (height: {avg_heights[cluster_label]})")
            items = ""

            # Get 3D point
            for item in cluster_items:
                detection = objectDetection()
                point3D = PointStamped(header=Header(frame_id=ARGS["CAMERA_FRAME"]), point=Point())

                box = item["bbox"]
                name = item["name"]
                score = item["score"]

                items += f"{name} "

                ymin, xmin, ymax, xmax = box
                if len(self.depth_image) != 0 and ARGS["DEPTH_ACTIVE"]:
                    
                    point2D = get2DCentroid(box, self.depth_image)
                    depth = get_depth(self.depth_image, point2D)

                    point3D_ = deproject_pixel_to_point(self.imageInfo, point2D, depth)
                    point3D.point.x = point3D_[0]
                    point3D.point.y = point3D_[1]
                    point3D.point.z = point3D_[2]   

                detection.labelText = name
                detection.score = score
                detection.label = cluster_label
                detection.ymin = ymin
                detection.xmin = xmin
                detection.ymax = ymax
                detection.xmax = xmax
                detection.point3D = point3D
                curr_level.detections.append(detection)

                # Draw bounding box for each detection
                (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
                (left, right, top, bottom) = (int(left * frame.shape[1]), int(right * frame.shape[1]),
                                            int(top * frame.shape[0]), int(bottom * frame.shape[0]))
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), colors[i], 2)
                cv2.putText(annotated_frame, f"{name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

            i += 1
            shelf_levels.levels.append(curr_level)
        
        print(f"Items: {items}")
        
        
        # Draw a line for average heights 
        i = 0
        for label, height in avg_heights.items():
            # y = int(height*annotated_frame.shape[0])
            y = int(height/len(clusters[label])*annotated_frame.shape[0])
            cv2.line(annotated_frame, (0, y), (annotated_frame.shape[1], y), colors[i], 2)
            cv2.putText(annotated_frame, f"{label}: {height:.2f}", (0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
            i += 1

        self.results_pub.publish(shelf_levels)
        self.detections_frame = annotated_frame
        self.visualize3D(shelf_levels)
        
    def visualize3D(self,levels):
        
        publish_marker_array = MarkerArray()
        i = 0

        for level in levels.levels:
            
            for detection in level.detections:
            
                marker = Marker()
                marker.color.a = 1.0
                marker.color.r = colors[i][0]
                marker.color.g = colors[i][1]
                marker.color.b = colors[i][2]
                marker.header.frame_id = ARGS["CAMERA_FRAME"]
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position = detection.point3D.point
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.lifetime = rospy.Duration(0.5)
                publish_marker_array.markers.append(marker)

            # print(f"Objects: {objects}")
  
        
        # print(f"Markers: {publish_marker_array}")
        self.results_3d_pub.publish(publish_marker_array)

    # Main function to run the detection model.
    def run(self, frame):
        frame_processed = frame
        detected_objects, visual_detections, visual_image = self.compute_result(frame_processed)
        self.cluster_objects(detected_objects, frame)

    
    def getHeights():
        pass

def main():
    for key in ARGS:
        ARGS[key] = rospy.get_param('~' + key, ARGS[key])
    ShelfDetection()

if __name__ == '__main__':
    main()