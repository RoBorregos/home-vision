;; Auto-generated. Do not edit!


(when (boundp 'vision::NewHost)
  (if (not (find-package "VISION"))
    (make-package "VISION"))
  (shadow 'NewHost (find-package "VISION")))
(unless (find-package "VISION::NEWHOST")
  (make-package "VISION::NEWHOST"))
(unless (find-package "VISION::NEWHOSTREQUEST")
  (make-package "VISION::NEWHOSTREQUEST"))
(unless (find-package "VISION::NEWHOSTRESPONSE")
  (make-package "VISION::NEWHOSTRESPONSE"))

(in-package "ROS")





(defclass vision::NewHostRequest
  :super ros::object
  :slots (_name ))

(defmethod vision::NewHostRequest
  (:init
   (&key
    ((:name __name) "")
    )
   (send-super :init)
   (setq _name (string __name))
   self)
  (:name
   (&optional __name)
   (if __name (setq _name __name)) _name)
  (:serialization-length
   ()
   (+
    ;; string _name
    4 (length _name)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _name
       (write-long (length _name) s) (princ _name s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass vision::NewHostResponse
  :super ros::object
  :slots (_success ))

(defmethod vision::NewHostResponse
  (:init
   (&key
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _success __success)
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass vision::NewHost
  :super ros::object
  :slots ())

(setf (get vision::NewHost :md5sum-) "d08a3b641c2f8680fbdfb1ea2e17a3e1")
(setf (get vision::NewHost :datatype-) "vision/NewHost")
(setf (get vision::NewHost :request) vision::NewHostRequest)
(setf (get vision::NewHost :response) vision::NewHostResponse)

(defmethod vision::NewHostRequest
  (:response () (instance vision::NewHostResponse :init)))

(setf (get vision::NewHostRequest :md5sum-) "d08a3b641c2f8680fbdfb1ea2e17a3e1")
(setf (get vision::NewHostRequest :datatype-) "vision/NewHostRequest")
(setf (get vision::NewHostRequest :definition-)
      "string name
---
bool success
")

(setf (get vision::NewHostResponse :md5sum-) "d08a3b641c2f8680fbdfb1ea2e17a3e1")
(setf (get vision::NewHostResponse :datatype-) "vision/NewHostResponse")
(setf (get vision::NewHostResponse :definition-)
      "string name
---
bool success
")



(provide :vision/NewHost "d08a3b641c2f8680fbdfb1ea2e17a3e1")


