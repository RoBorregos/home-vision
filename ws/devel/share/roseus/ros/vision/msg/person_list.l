;; Auto-generated. Do not edit!


(when (boundp 'vision::person_list)
  (if (not (find-package "VISION"))
    (make-package "VISION"))
  (shadow 'person_list (find-package "VISION")))
(unless (find-package "VISION::PERSON_LIST")
  (make-package "VISION::PERSON_LIST"))

(in-package "ROS")
;;//! \htmlinclude person_list.msg.html


(defclass vision::person_list
  :super ros::object
  :slots (_list ))

(defmethod vision::person_list
  (:init
   (&key
    ((:list __list) ())
    )
   (send-super :init)
   (setq _list __list)
   self)
  (:list
   (&rest __list)
   (if (keywordp (car __list))
       (send* _list __list)
     (progn
       (if __list (setq _list (car __list)))
       _list)))
  (:serialization-length
   ()
   (+
    ;; vision/person[] _list
    (apply #'+ (send-all _list :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; vision/person[] _list
     (write-long (length _list) s)
     (dolist (elem _list)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; vision/person[] _list
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _list (let (r) (dotimes (i n) (push (instance vision::person :init) r)) r))
     (dolist (elem- _list)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get vision::person_list :md5sum-) "6e0e3b7caba85042fa0a8abdf1c715af")
(setf (get vision::person_list :datatype-) "vision/person_list")
(setf (get vision::person_list :definition-)
      "vision/person[] list
================================================================================
MSG: vision/person
string name
int64 x
int64 y
")



(provide :vision/person_list "6e0e3b7caba85042fa0a8abdf1c715af")


