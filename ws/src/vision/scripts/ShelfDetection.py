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
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Point, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from vision.msg import objectDetection, level, shelf
import numpy as np


# ARGS= {
#     "SOURCE": SOURCES["VIDEO"],
#     "ROS_INPUT": False,
#     "USE_ACTIVE_FLAG": True,
#     "DEPTH_ACTIVE": False,
#     "DEPTH_INPUT": "/camera/depth/image_raw",
#     "CAMERA_INFO": "/camera/depth/camera_info",
#     "MIN_SCORE_THRESH": 0.5,
#     "VERBOSE": True,
#     "CAMERA_FRAME": "xtion_rgb_optical_frame",
#     "YOLO_MODEL_PATH": str(pathlib.Path(__file__).parent) + "/../models/yolov5s.pt",
#     "FLIP_IMAGE": False,
# }

CAMERA_TOPIC = "/zed2/zed_node/rgb/image_rect_color"  
FLIP_IMAGE = False  
MIN_SCORE_THRESH = 0.5
VERBOSE = True
MIN_CLUSTERS = 2
MAX_CLUSTERS = 6
RESULTS_TOPIC = "/shelf_detection"
ACTIVE_SERVICE_TOPIC = "/shelf_detection_active"
CAMERA_FRAME = "zed2_rgb_optical_frame"

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
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback)
        self.results_pub = rospy.Publisher(RESULTS_TOPIC, shelf, queue_size=1)
        self.detections_frame = []
        self.model = YOLO('yolov8n.pt')
        # self.model = YOLO('yolov5s.pt')
        # self.image = None

        def yolov8_warmup(model, repetitions=1, verbose=False):
            # Warmup model
            startTime = time.time()
            # create an empty frame to warmup the model
            for i in range(repetitions):
                warmupFrame = np.zeros((360, 640, 3), dtype=np.uint8)
                model.predict(source=warmupFrame, verbose=verbose)
            rospy.logdebug(f"Model warmed up in {time.time() - startTime} seconds")

        def loadYolov8():
            self.model = YOLO('yolov8n.pt')


        # loadYolov8()

        self.detections_frame = []
        try:
            rate = rospy.Rate(60)

            while not rospy.is_shutdown():
                if VERBOSE and len(self.detections_frame) != 0:
                # if VERBOSE and self.detections_frame != None:
                    cv2.imshow("Detections", self.detections_frame)
                    cv2.waitKey(1)

                # if len(self.detections_frame) != 0:
                #     self.image_publisher.publish(self.bridge.cv2_to_imgmsg(self.detections_frame, "bgr8"))
                    
                rate.sleep()
        except KeyboardInterrupt:
            pass

        cv2.destroyAllWindows()

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.run(frame)

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
            if conf < MIN_SCORE_THRESH:
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
                if prob < MIN_SCORE_THRESH:
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
    
    # def run_inference_on_image(self, frame):
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frame_np = np.expand_dims(frame, axis=0)
    #     input_tensor = tf.convert_to_tensor(frame_np, dtype=tf.uint8)

    #     if ARGS["VERBOSE"]:
    #         print('Predicting...')

    #     start_time = time.time()
    #     detections = self.detect_fn(input_tensor)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
        
    #     if ARGS["VERBOSE"]:
    #         print('Done! Took {} seconds'.format(elapsed_time))

    #     num_detections = int(detections.pop('num_detections'))
    #     detections = {key: value[0, :num_detections].numpy()
    #                 for key, value in detections.items()}
    #     detections['num_detections'] = num_detections

    #     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    #     output = {
    #         'detection_boxes': detections['detection_boxes'],
    #         'detection_classes': detections['detection_classes'],
    #         'detection_scores': detections['detection_scores']
    #     }
    #     return output
    
    
    def get_objects(self, boxes, scores, classes, names, height, width, frame):
        objects = {}
        res = []

        pa = PoseArray()
        pa.header.frame_id = ARGS["CAMERA_FRAME"]
        pa.header.stamp = rospy.Time.now()

        for index, value in enumerate(classes):
            if scores[index] > ARGS["MIN_SCORE_THRESH"]:
                if value in objects:
                    # in case it detects more that one of each object, grabs the one with higher score
                    if objects[value]['score'] > scores[index]:
                        continue
                
                point3D = PointStamped(header=Header(frame_id=ARGS["CAMERA_FRAME"]), point=Point())

                if ARGS["DEPTH_ACTIVE"] and len(self.depth_image) != 0:
                    # if frame is flipped, flip the point2D
                    point2D = self.get2DCentroid(boxes[index], self.depth_image)
                    #rospy.loginfo("Point2D: " + str(point2D))
                    depth = get_depth(self.depth_image, point2D) ## in m
                    #rospy.loginfo("Depth: " + str(depth))
                    #depth = depth / 1000 ## in mm
                    point3D_ = deproject_pixel_to_point(self.imageInfo, point2D, depth)
                    #rospy.loginfo("Point3D: " + str(point3D_))
                    point3D.point.x = point3D_[0]
                    point3D.point.y = point3D_[1]
                    point3D.point.z = point3D_[2]
                    pa.poses.append(Pose(position=point3D.point))

                objects[value] = {
                    "name": names[index],
                    "score": float(scores[index]),
                    "ymin": float(boxes[index][0]),
                    "xmin": float(boxes[index][1]),
                    "ymax": float(boxes[index][2]),
                    "xmax": float(boxes[index][3]),
                    "centroid_x": point2D[0],
                    "centroid_y": point2D[1],
                    "point3D": point3D
                }
        # self.posePublisher.publish(pa)
        
        for label in objects:
            labelText = objects[label]["name"]
            detection = objects[label]
            res.append(objectDetection(
                    label = int(label),
                    labelText = str(labelText),
                    score = detection["score"],
                    ymin =  detection["ymin"],
                    xmin =  detection["xmin"],
                    ymax =  detection["ymax"],
                    xmax =  detection["xmax"],
                    point3D = detection["point3D"]
                ))
            
        # visualize here
        publish_marker_array = MarkerArray()
        
        for i, label in enumerate(objects):
            detection = objects[label]
            # generate markers for each object
            marker = Marker()
            marker.header.frame_id = ARGS["CAMERA_FRAME"]
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = detection["point3D"].point
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.lifetime = rospy.Duration(0.5)
            publish_marker_array.markers.append(marker)
        
        # print(f"Markers: {publish_marker_array}")
        self.objects_publisher_3d.publish(publish_marker_array)
        
            # publish 
        return res
    
        # Handle the detection model input/output.
    def compute_result(self, frame):
        visual_frame = copy.deepcopy(frame)
        # visual_detections = self.yolo_run_inference_on_image(visual_frame)
        visual_detections = self.yolov8_run_inference_on_image(visual_frame)

        detections = copy.deepcopy(visual_detections)

        detections["detection_boxes"] = np.array([[1 - box[2], 1 - box[3], 1 - box[0], 1 - box[1]] for box in visual_detections["detection_boxes"]]) if FLIP_IMAGE else visual_detections["detection_boxes"]
        
        return detections, visual_detections, visual_frame
        # return self.get_objects(detections["detection_boxes"],
        #                         detections["detection_scores"],
        #                         detections["detection_classes"],
        #                         detections["detection_names"],
        #                         frame.shape[0],
        #                         frame.shape[1],
        #                         frame), visual_detections, visual_frame

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

        boxes = detected_objects["detection_boxes"]
        names = detected_objects["detection_names"]
        scores = detected_objects["detection_scores"]

        annotated_frame = self.visualize_detections(
            frame,
            detected_objects['detection_boxes'],
            detected_objects['detection_classes'],
            detected_objects["detection_names"],
            detected_objects['detection_scores'],
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=MIN_SCORE_THRESH,
            agnostic_mode=False)
        
        self.detections_frame = annotated_frame
         
        indexes = {}

        for idx, box in enumerate(boxes):
            ymin, xmin, ymax, xmax = box
            y_mid.append((ymin + ymax) / 2)
            y_low.append(ymax)
            indexes[ymin] = idx

        
        if len(y_low) < 2:
            print("Not enough objects to cluster")
            return

        y_low_array = np.array(y_low).reshape(-1, 1)
        
        optima_clusters = None
        optimal_cluster_labels = None
        max_silhouette_score = -1

        max_clusters = min(MAX_CLUSTERS, len(y_low)-1)


        # Determine the optimal number of clusters
        for n_clusters in range(MIN_CLUSTERS, max_clusters+1):
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            kmeans.fit(y_low_array)
            cluster_labels = kmeans.labels_
            silhouette_avg = silhouette_score(y_low_array, cluster_labels)
            # print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")

            if silhouette_avg > max_silhouette_score:
                max_silhouette_score = silhouette_avg
                optima_clusters = n_clusters
                optimal_cluster_labels = cluster_labels

        if optima_clusters is None:
            print("No optimal number of clusters found")
            return
        
        # Group objects by cluster
        clusters = {}
        avg_heights = {}

        
        for idx, label in enumerate(optimal_cluster_labels):
            if label not in clusters:
                clusters[label] = []
                avg_heights[label] = 0
            # y_min = y_low[idx]
            clusters[label].append({
                "name": names[idx],
                "bbox": boxes[idx],
                "score": scores[idx]
                })
            
            # avg_heights[label] += y_low[idx] 
            avg_heights[label] += y_mid[idx] 
            
            # print(names[idx])

        # Print the clusters
        print(f"Optimal number of clusters: {optima_clusters}")
        annotated_frame = frame

        i = 0

        levels = shelf()
        levels.levels = []
        for cluster_label, cluster_items in clusters.items():
            curr_level = level()
            curr_level.label = cluster_label
            curr_level.mid_height = avg_heights[cluster_label]/len(clusters[cluster_label])

            
            # Draw bbox
            print(f"Cluster {cluster_label}: height: {avg_heights[cluster_label]} {cluster_items}")

            for item in cluster_items:
                detection = objectDetection()
                point3D = PointStamped(header=Header(frame_id=CAMERA_FRAME), point=Point())

                box = item["bbox"]
                name = item["name"]
                score = item["score"]

                ymin, xmin, ymax, xmax = box
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



                (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
                (left, right, top, bottom) = (int(left * frame.shape[1]), int(right * frame.shape[1]),
                                            int(top * frame.shape[0]), int(bottom * frame.shape[0]))
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), colors[i], 2)
                cv2.putText(annotated_frame, f"{name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

            i += 1
            levels.levels.append(curr_level)
            
        

        
        
        # Draw a line for average heights 
        i = 0
        for label, height in avg_heights.items():
            # y = int(height*annotated_frame.shape[0])
            y = int(height/len(clusters[label])*annotated_frame.shape[0])
            print(height, y, frame.shape[0])
            cv2.line(annotated_frame, (0, y), (annotated_frame.shape[1], y), colors[i], 2)
            cv2.putText(annotated_frame, f"{label}: {height:.2f}", (0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
            i += 1
        # for idx, label in enumerate(optimal_cluster_labels):
        #     if label in avg_heights:
        #         ymin, xmin, ymax, xmax = boxes[idx]
        #         y = int((ymin + ymax) / 2)
        #         cv2.line(annotated_frame, (0, y), (annotated_frame.shape[1], y), (0, 255, 255), 2)
        #         cv2.putText(annotated_frame, f"{avg_heights[label]:.2f}", (0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        self.results_pub.publish(shelf)
        self.detections_frame = annotated_frame
        # for detected in detected_objects:
        #     print(detected["detection_boxes"])
            # ymin = detected[""]
            # ymin = detected.ymin
            # ymax = detected.ymax
            # y_mid.append((ymin + ymax) / 2)
            # y_low.append(ymin)

        # y_coordinates = [1.2, 2.4, 1.8, 3.5, 4.2, 2.0, 3.0, 4.5]

        # # Reshape data into a numpy array (required for KMeans)
        # y_array = np.array(y_coordinates).reshape(-1, 1)

        # # Define the number of clusters (groups)
        # num_clusters = 2

        # # Initialize KMeans clustering algorithm
        # kmeans = KMeans(n_clusters=num_clusters)

        # # Fit the KMeans model to the data
        # kmeans.fit(y_array)

        # # Get the cluster labels for each data point
        # cluster_labels = kmeans.labels_

        # # Create a dictionary to store objects grouped by cluster
        # clusters = {}
        # for idx, label in enumerate(cluster_labels):
        #     if label not in clusters:
        #         clusters[label] = []
        #     clusters[label].append(y_coordinates[idx])

        # # Print the clusters
        # for cluster_label, cluster_items in clusters.items():
        #     print(f"Cluster {cluster_label}: {cluster_items}")

        # print("Classifying shelves")


        # Main function to run the detection model.
    def run(self, frame):
        frame_processed = frame
        # frame_processed = imutils.resize(frame, width=500)

        

        detected_objects, visual_detections, visual_image = self.compute_result(frame_processed)
        self.cluster_objects(detected_objects, frame)
        # self.compute_result(frame_processed)

        frame = self.visualize_detections(
            visual_image,
            visual_detections['detection_boxes'],
            visual_detections['detection_classes'],
            visual_detections["detection_names"],
            visual_detections['detection_scores'],
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=MIN_SCORE_THRESH,
            agnostic_mode=False)

        # self.detections_frame = frame

        #print("PUBLISHED DATA")
        # self.publisher.publish(objectDetectionArray(detections=detected_objects))
        # self.fps.update()

    
    def getHeights():
        pass

def main():
    # for key in ARGS:
    #     ARGS[key] = rospy.get_param('~' + key, ARGS[key])
    ShelfDetection()

if __name__ == '__main__':
    main()