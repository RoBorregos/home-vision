;; Auto-generated. Do not edit!


(when (boundp 'vision::people_count)
  (if (not (find-package "VISION"))
    (make-package "VISION"))
  (shadow 'people_count (find-package "VISION")))
(unless (find-package "VISION::PEOPLE_COUNT")
  (make-package "VISION::PEOPLE_COUNT"))

(in-package "ROS")
;;//! \htmlinclude people_count.msg.html


(defclass vision::people_count
  :super ros::object
  :slots (_detected_people _people_standing _people_sitting _people_raising_hand _people_pointing ))

(defmethod vision::people_count
  (:init
   (&key
    ((:detected_people __detected_people) 0)
    ((:people_standing __people_standing) 0)
    ((:people_sitting __people_sitting) 0)
    ((:people_raising_hand __people_raising_hand) 0)
    ((:people_pointing __people_pointing) 0)
    )
   (send-super :init)
   (setq _detected_people (round __detected_people))
   (setq _people_standing (round __people_standing))
   (setq _people_sitting (round __people_sitting))
   (setq _people_raising_hand (round __people_raising_hand))
   (setq _people_pointing (round __people_pointing))
   self)
  (:detected_people
   (&optional __detected_people)
   (if __detected_people (setq _detected_people __detected_people)) _detected_people)
  (:people_standing
   (&optional __people_standing)
   (if __people_standing (setq _people_standing __people_standing)) _people_standing)
  (:people_sitting
   (&optional __people_sitting)
   (if __people_sitting (setq _people_sitting __people_sitting)) _people_sitting)
  (:people_raising_hand
   (&optional __people_raising_hand)
   (if __people_raising_hand (setq _people_raising_hand __people_raising_hand)) _people_raising_hand)
  (:people_pointing
   (&optional __people_pointing)
   (if __people_pointing (setq _people_pointing __people_pointing)) _people_pointing)
  (:serialization-length
   ()
   (+
    ;; int64 _detected_people
    8
    ;; int64 _people_standing
    8
    ;; int64 _people_sitting
    8
    ;; int64 _people_raising_hand
    8
    ;; int64 _people_pointing
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int64 _detected_people
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _detected_people (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _detected_people) (= (length (_detected_people . bv)) 2)) ;; bignum
              (write-long (ash (elt (_detected_people . bv) 0) 0) s)
              (write-long (ash (elt (_detected_people . bv) 1) -1) s))
             ((and (class _detected_people) (= (length (_detected_people . bv)) 1)) ;; big1
              (write-long (elt (_detected_people . bv) 0) s)
              (write-long (if (>= _detected_people 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _detected_people s)(write-long (if (>= _detected_people 0) 0 #xffffffff) s)))
     ;; int64 _people_standing
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _people_standing (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _people_standing) (= (length (_people_standing . bv)) 2)) ;; bignum
              (write-long (ash (elt (_people_standing . bv) 0) 0) s)
              (write-long (ash (elt (_people_standing . bv) 1) -1) s))
             ((and (class _people_standing) (= (length (_people_standing . bv)) 1)) ;; big1
              (write-long (elt (_people_standing . bv) 0) s)
              (write-long (if (>= _people_standing 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _people_standing s)(write-long (if (>= _people_standing 0) 0 #xffffffff) s)))
     ;; int64 _people_sitting
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _people_sitting (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _people_sitting) (= (length (_people_sitting . bv)) 2)) ;; bignum
              (write-long (ash (elt (_people_sitting . bv) 0) 0) s)
              (write-long (ash (elt (_people_sitting . bv) 1) -1) s))
             ((and (class _people_sitting) (= (length (_people_sitting . bv)) 1)) ;; big1
              (write-long (elt (_people_sitting . bv) 0) s)
              (write-long (if (>= _people_sitting 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _people_sitting s)(write-long (if (>= _people_sitting 0) 0 #xffffffff) s)))
     ;; int64 _people_raising_hand
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _people_raising_hand (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _people_raising_hand) (= (length (_people_raising_hand . bv)) 2)) ;; bignum
              (write-long (ash (elt (_people_raising_hand . bv) 0) 0) s)
              (write-long (ash (elt (_people_raising_hand . bv) 1) -1) s))
             ((and (class _people_raising_hand) (= (length (_people_raising_hand . bv)) 1)) ;; big1
              (write-long (elt (_people_raising_hand . bv) 0) s)
              (write-long (if (>= _people_raising_hand 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _people_raising_hand s)(write-long (if (>= _people_raising_hand 0) 0 #xffffffff) s)))
     ;; int64 _people_pointing
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _people_pointing (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _people_pointing) (= (length (_people_pointing . bv)) 2)) ;; bignum
              (write-long (ash (elt (_people_pointing . bv) 0) 0) s)
              (write-long (ash (elt (_people_pointing . bv) 1) -1) s))
             ((and (class _people_pointing) (= (length (_people_pointing . bv)) 1)) ;; big1
              (write-long (elt (_people_pointing . bv) 0) s)
              (write-long (if (>= _people_pointing 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _people_pointing s)(write-long (if (>= _people_pointing 0) 0 #xffffffff) s)))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int64 _detected_people
#+(or :alpha :irix6 :x86_64)
      (setf _detected_people (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _detected_people (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;; int64 _people_standing
#+(or :alpha :irix6 :x86_64)
      (setf _people_standing (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _people_standing (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;; int64 _people_sitting
#+(or :alpha :irix6 :x86_64)
      (setf _people_sitting (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _people_sitting (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;; int64 _people_raising_hand
#+(or :alpha :irix6 :x86_64)
      (setf _people_raising_hand (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _people_raising_hand (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;; int64 _people_pointing
#+(or :alpha :irix6 :x86_64)
      (setf _people_pointing (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _people_pointing (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
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

(setf (get vision::people_count :md5sum-) "dd5d4b83e54fd0abe744ecf17478c695")
(setf (get vision::people_count :datatype-) "vision/people_count")
(setf (get vision::people_count :definition-)
      "int64 detected_people
int64 people_standing
int64 people_sitting
int64 people_raising_hand
int64 people_pointing
")



(provide :vision/people_count "dd5d4b83e54fd0abe744ecf17478c695")


