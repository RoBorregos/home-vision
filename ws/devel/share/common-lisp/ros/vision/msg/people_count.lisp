; Auto-generated. Do not edit!


(cl:in-package vision-msg)


;//! \htmlinclude people_count.msg.html

(cl:defclass <people_count> (roslisp-msg-protocol:ros-message)
  ((detected_people
    :reader detected_people
    :initarg :detected_people
    :type cl:integer
    :initform 0)
   (people_standing
    :reader people_standing
    :initarg :people_standing
    :type cl:integer
    :initform 0)
   (people_sitting
    :reader people_sitting
    :initarg :people_sitting
    :type cl:integer
    :initform 0)
   (people_raising_hand
    :reader people_raising_hand
    :initarg :people_raising_hand
    :type cl:integer
    :initform 0)
   (people_pointing
    :reader people_pointing
    :initarg :people_pointing
    :type cl:integer
    :initform 0))
)

(cl:defclass people_count (<people_count>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <people_count>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'people_count)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vision-msg:<people_count> is deprecated: use vision-msg:people_count instead.")))

(cl:ensure-generic-function 'detected_people-val :lambda-list '(m))
(cl:defmethod detected_people-val ((m <people_count>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision-msg:detected_people-val is deprecated.  Use vision-msg:detected_people instead.")
  (detected_people m))

(cl:ensure-generic-function 'people_standing-val :lambda-list '(m))
(cl:defmethod people_standing-val ((m <people_count>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision-msg:people_standing-val is deprecated.  Use vision-msg:people_standing instead.")
  (people_standing m))

(cl:ensure-generic-function 'people_sitting-val :lambda-list '(m))
(cl:defmethod people_sitting-val ((m <people_count>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision-msg:people_sitting-val is deprecated.  Use vision-msg:people_sitting instead.")
  (people_sitting m))

(cl:ensure-generic-function 'people_raising_hand-val :lambda-list '(m))
(cl:defmethod people_raising_hand-val ((m <people_count>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision-msg:people_raising_hand-val is deprecated.  Use vision-msg:people_raising_hand instead.")
  (people_raising_hand m))

(cl:ensure-generic-function 'people_pointing-val :lambda-list '(m))
(cl:defmethod people_pointing-val ((m <people_count>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision-msg:people_pointing-val is deprecated.  Use vision-msg:people_pointing instead.")
  (people_pointing m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <people_count>) ostream)
  "Serializes a message object of type '<people_count>"
  (cl:let* ((signed (cl:slot-value msg 'detected_people)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'people_standing)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'people_sitting)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'people_raising_hand)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'people_pointing)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <people_count>) istream)
  "Deserializes a message object of type '<people_count>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'detected_people) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'people_standing) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'people_sitting) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'people_raising_hand) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'people_pointing) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<people_count>)))
  "Returns string type for a message object of type '<people_count>"
  "vision/people_count")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'people_count)))
  "Returns string type for a message object of type 'people_count"
  "vision/people_count")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<people_count>)))
  "Returns md5sum for a message object of type '<people_count>"
  "dd5d4b83e54fd0abe744ecf17478c695")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'people_count)))
  "Returns md5sum for a message object of type 'people_count"
  "dd5d4b83e54fd0abe744ecf17478c695")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<people_count>)))
  "Returns full string definition for message of type '<people_count>"
  (cl:format cl:nil "int64 detected_people~%int64 people_standing~%int64 people_sitting~%int64 people_raising_hand~%int64 people_pointing~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'people_count)))
  "Returns full string definition for message of type 'people_count"
  (cl:format cl:nil "int64 detected_people~%int64 people_standing~%int64 people_sitting~%int64 people_raising_hand~%int64 people_pointing~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <people_count>))
  (cl:+ 0
     8
     8
     8
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <people_count>))
  "Converts a ROS message object to a list"
  (cl:list 'people_count
    (cl:cons ':detected_people (detected_people msg))
    (cl:cons ':people_standing (people_standing msg))
    (cl:cons ':people_sitting (people_sitting msg))
    (cl:cons ':people_raising_hand (people_raising_hand msg))
    (cl:cons ':people_pointing (people_pointing msg))
))
