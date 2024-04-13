;; Auto-generated. Do not edit!


(when (boundp 'vision::PersonCount)
  (if (not (find-package "VISION"))
    (make-package "VISION"))
  (shadow 'PersonCount (find-package "VISION")))
(unless (find-package "VISION::PERSONCOUNT")
  (make-package "VISION::PERSONCOUNT"))
(unless (find-package "VISION::PERSONCOUNTREQUEST")
  (make-package "VISION::PERSONCOUNTREQUEST"))
(unless (find-package "VISION::PERSONCOUNTRESPONSE")
  (make-package "VISION::PERSONCOUNTRESPONSE"))

(in-package "ROS")





(defclass vision::PersonCountRequest
  :super ros::object
  :slots (_data ))

(defmethod vision::PersonCountRequest
  (:init
   (&key
    ((:data __data) "")
    )
   (send-super :init)
   (setq _data (string __data))
   self)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; string _data
    4 (length _data)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _data
       (write-long (length _data) s) (princ _data s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _data
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _data (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass vision::PersonCountResponse
  :super ros::object
  :slots (_count ))

(defmethod vision::PersonCountResponse
  (:init
   (&key
    ((:count __count) 0)
    )
   (send-super :init)
   (setq _count (round __count))
   self)
  (:count
   (&optional __count)
   (if __count (setq _count __count)) _count)
  (:serialization-length
   ()
   (+
    ;; int64 _count
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int64 _count
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _count (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _count) (= (length (_count . bv)) 2)) ;; bignum
              (write-long (ash (elt (_count . bv) 0) 0) s)
              (write-long (ash (elt (_count . bv) 1) -1) s))
             ((and (class _count) (= (length (_count . bv)) 1)) ;; big1
              (write-long (elt (_count . bv) 0) s)
              (write-long (if (>= _count 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _count s)(write-long (if (>= _count 0) 0 #xffffffff) s)))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int64 _count
#+(or :alpha :irix6 :x86_64)
      (setf _count (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _count (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;;
   self)
  )

(defclass vision::PersonCount
  :super ros::object
  :slots ())

(setf (get vision::PersonCount :md5sum-) "5c1a05469ceca6f2dc82e0bc5828de17")
(setf (get vision::PersonCount :datatype-) "vision/PersonCount")
(setf (get vision::PersonCount :request) vision::PersonCountRequest)
(setf (get vision::PersonCount :response) vision::PersonCountResponse)

(defmethod vision::PersonCountRequest
  (:response () (instance vision::PersonCountResponse :init)))

(setf (get vision::PersonCountRequest :md5sum-) "5c1a05469ceca6f2dc82e0bc5828de17")
(setf (get vision::PersonCountRequest :datatype-) "vision/PersonCountRequest")
(setf (get vision::PersonCountRequest :definition-)
      "string data
---
int64 count
")

(setf (get vision::PersonCountResponse :md5sum-) "5c1a05469ceca6f2dc82e0bc5828de17")
(setf (get vision::PersonCountResponse :datatype-) "vision/PersonCountResponse")
(setf (get vision::PersonCountResponse :definition-)
      "string data
---
int64 count
")



(provide :vision/PersonCount "5c1a05469ceca6f2dc82e0bc5828de17")


