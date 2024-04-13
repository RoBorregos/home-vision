; Auto-generated. Do not edit!


(cl:in-package vision-srv)


;//! \htmlinclude NewHost-request.msg.html

(cl:defclass <NewHost-request> (roslisp-msg-protocol:ros-message)
  ((name
    :reader name
    :initarg :name
    :type cl:string
    :initform ""))
)

(cl:defclass NewHost-request (<NewHost-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <NewHost-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'NewHost-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vision-srv:<NewHost-request> is deprecated: use vision-srv:NewHost-request instead.")))

(cl:ensure-generic-function 'name-val :lambda-list '(m))
(cl:defmethod name-val ((m <NewHost-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision-srv:name-val is deprecated.  Use vision-srv:name instead.")
  (name m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <NewHost-request>) ostream)
  "Serializes a message object of type '<NewHost-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'name))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <NewHost-request>) istream)
  "Deserializes a message object of type '<NewHost-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<NewHost-request>)))
  "Returns string type for a service object of type '<NewHost-request>"
  "vision/NewHostRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'NewHost-request)))
  "Returns string type for a service object of type 'NewHost-request"
  "vision/NewHostRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<NewHost-request>)))
  "Returns md5sum for a message object of type '<NewHost-request>"
  "d08a3b641c2f8680fbdfb1ea2e17a3e1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'NewHost-request)))
  "Returns md5sum for a message object of type 'NewHost-request"
  "d08a3b641c2f8680fbdfb1ea2e17a3e1")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<NewHost-request>)))
  "Returns full string definition for message of type '<NewHost-request>"
  (cl:format cl:nil "string name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'NewHost-request)))
  "Returns full string definition for message of type 'NewHost-request"
  (cl:format cl:nil "string name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <NewHost-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'name))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <NewHost-request>))
  "Converts a ROS message object to a list"
  (cl:list 'NewHost-request
    (cl:cons ':name (name msg))
))
;//! \htmlinclude NewHost-response.msg.html

(cl:defclass <NewHost-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass NewHost-response (<NewHost-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <NewHost-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'NewHost-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vision-srv:<NewHost-response> is deprecated: use vision-srv:NewHost-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <NewHost-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision-srv:success-val is deprecated.  Use vision-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <NewHost-response>) ostream)
  "Serializes a message object of type '<NewHost-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <NewHost-response>) istream)
  "Deserializes a message object of type '<NewHost-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<NewHost-response>)))
  "Returns string type for a service object of type '<NewHost-response>"
  "vision/NewHostResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'NewHost-response)))
  "Returns string type for a service object of type 'NewHost-response"
  "vision/NewHostResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<NewHost-response>)))
  "Returns md5sum for a message object of type '<NewHost-response>"
  "d08a3b641c2f8680fbdfb1ea2e17a3e1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'NewHost-response)))
  "Returns md5sum for a message object of type 'NewHost-response"
  "d08a3b641c2f8680fbdfb1ea2e17a3e1")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<NewHost-response>)))
  "Returns full string definition for message of type '<NewHost-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'NewHost-response)))
  "Returns full string definition for message of type 'NewHost-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <NewHost-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <NewHost-response>))
  "Converts a ROS message object to a list"
  (cl:list 'NewHost-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'NewHost)))
  'NewHost-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'NewHost)))
  'NewHost-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'NewHost)))
  "Returns string type for a service object of type '<NewHost>"
  "vision/NewHost")