;; Auto-generated. Do not edit!


(when (boundp 'vision::img_list)
  (if (not (find-package "VISION"))
    (make-package "VISION"))
  (shadow 'img_list (find-package "VISION")))
(unless (find-package "VISION::IMG_LIST")
  (make-package "VISION::IMG_LIST"))

(in-package "ROS")
;;//! \htmlinclude img_list.msg.html


(defclass vision::img_list
  :super ros::object
  :slots (_images ))

(defmethod vision::img_list
  (:init
   (&key
    ((:images __images) ())
    )
   (send-super :init)
   (setq _images __images)
   self)
  (:images
   (&rest __images)
   (if (keywordp (car __images))
       (send* _images __images)
     (progn
       (if __images (setq _images (car __images)))
       _images)))
  (:serialization-length
   ()
   (+
    ;; vision/img[] _images
    (apply #'+ (send-all _images :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; vision/img[] _images
     (write-long (length _images) s)
     (dolist (elem _images)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; vision/img[] _images
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _images (let (r) (dotimes (i n) (push (instance vision::img :init) r)) r))
     (dolist (elem- _images)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get vision::img_list :md5sum-) "beeae6a30f35d07ad6b429e4f70ceea5")
(setf (get vision::img_list :datatype-) "vision/img_list")
(setf (get vision::img_list :definition-)
      "vision/img[] images
================================================================================
MSG: vision/img
int64 x
int64 y
int64 w
int64 h
string name
")



(provide :vision/img_list "beeae6a30f35d07ad6b429e4f70ceea5")


