; Auto-generated. Do not edit!


(cl:in-package vision-srv)


;//! \htmlinclude PersonCount-request.msg.html

(cl:defclass <PersonCount-request> (roslisp-msg-protocol:ros-message)
  ((data
    :reader data
    :initarg :data
    :type cl:string
    :initform ""))
)

(cl:defclass PersonCount-request (<PersonCount-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PersonCount-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PersonCount-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vision-srv:<PersonCount-request> is deprecated: use vision-srv:PersonCount-request instead.")))

(cl:ensure-generic-function 'data-val :lambda-list '(m))
(cl:defmethod data-val ((m <PersonCount-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision-srv:data-val is deprecated.  Use vision-srv:data instead.")
  (data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PersonCount-request>) ostream)
  "Serializes a message object of type '<PersonCount-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'data))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'data))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PersonCount-request>) istream)
  "Deserializes a message object of type '<PersonCount-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'data) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'data) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PersonCount-request>)))
  "Returns string type for a service object of type '<PersonCount-request>"
  "vision/PersonCountRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PersonCount-request)))
  "Returns string type for a service object of type 'PersonCount-request"
  "vision/PersonCountRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PersonCount-request>)))
  "Returns md5sum for a message object of type '<PersonCount-request>"
  "5c1a05469ceca6f2dc82e0bc5828de17")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PersonCount-request)))
  "Returns md5sum for a message object of type 'PersonCount-request"
  "5c1a05469ceca6f2dc82e0bc5828de17")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PersonCount-request>)))
  "Returns full string definition for message of type '<PersonCount-request>"
  (cl:format cl:nil "string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PersonCount-request)))
  "Returns full string definition for message of type 'PersonCount-request"
  (cl:format cl:nil "string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PersonCount-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'data))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PersonCount-request>))
  "Converts a ROS message object to a list"
  (cl:list 'PersonCount-request
    (cl:cons ':data (data msg))
))
;//! \htmlinclude PersonCount-response.msg.html

(cl:defclass <PersonCount-response> (roslisp-msg-protocol:ros-message)
  ((count
    :reader count
    :initarg :count
    :type cl:integer
    :initform 0))
)

(cl:defclass PersonCount-response (<PersonCount-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PersonCount-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PersonCount-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vision-srv:<PersonCount-response> is deprecated: use vision-srv:PersonCount-response instead.")))

(cl:ensure-generic-function 'count-val :lambda-list '(m))
(cl:defmethod count-val ((m <PersonCount-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision-srv:count-val is deprecated.  Use vision-srv:count instead.")
  (count m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PersonCount-response>) ostream)
  "Serializes a message object of type '<PersonCount-response>"
  (cl:let* ((signed (cl:slot-value msg 'count)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
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
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PersonCount-response>) istream)
  "Deserializes a message object of type '<PersonCount-response>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'count) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PersonCount-response>)))
  "Returns string type for a service object of type '<PersonCount-response>"
  "vision/PersonCountResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PersonCount-response)))
  "Returns string type for a service object of type 'PersonCount-response"
  "vision/PersonCountResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PersonCount-response>)))
  "Returns md5sum for a message object of type '<PersonCount-response>"
  "5c1a05469ceca6f2dc82e0bc5828de17")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PersonCount-response)))
  "Returns md5sum for a message object of type 'PersonCount-response"
  "5c1a05469ceca6f2dc82e0bc5828de17")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PersonCount-response>)))
  "Returns full string definition for message of type '<PersonCount-response>"
  (cl:format cl:nil "int64 count~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PersonCount-response)))
  "Returns full string definition for message of type 'PersonCount-response"
  (cl:format cl:nil "int64 count~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PersonCount-response>))
  (cl:+ 0
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PersonCount-response>))
  "Converts a ROS message object to a list"
  (cl:list 'PersonCount-response
    (cl:cons ':count (count msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'PersonCount)))
  'PersonCount-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'PersonCount)))
  'PersonCount-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PersonCount)))
  "Returns string type for a service object of type '<PersonCount>"
  "vision/PersonCount")