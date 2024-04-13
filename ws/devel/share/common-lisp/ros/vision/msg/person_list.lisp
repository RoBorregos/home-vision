; Auto-generated. Do not edit!


(cl:in-package vision-msg)


;//! \htmlinclude person_list.msg.html

(cl:defclass <person_list> (roslisp-msg-protocol:ros-message)
  ((list
    :reader list
    :initarg :list
    :type (cl:vector vision-msg:person)
   :initform (cl:make-array 0 :element-type 'vision-msg:person :initial-element (cl:make-instance 'vision-msg:person))))
)

(cl:defclass person_list (<person_list>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <person_list>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'person_list)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vision-msg:<person_list> is deprecated: use vision-msg:person_list instead.")))

(cl:ensure-generic-function 'list-val :lambda-list '(m))
(cl:defmethod list-val ((m <person_list>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision-msg:list-val is deprecated.  Use vision-msg:list instead.")
  (list m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <person_list>) ostream)
  "Serializes a message object of type '<person_list>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'list))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'list))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <person_list>) istream)
  "Deserializes a message object of type '<person_list>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'list) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'list)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'vision-msg:person))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<person_list>)))
  "Returns string type for a message object of type '<person_list>"
  "vision/person_list")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'person_list)))
  "Returns string type for a message object of type 'person_list"
  "vision/person_list")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<person_list>)))
  "Returns md5sum for a message object of type '<person_list>"
  "6e0e3b7caba85042fa0a8abdf1c715af")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'person_list)))
  "Returns md5sum for a message object of type 'person_list"
  "6e0e3b7caba85042fa0a8abdf1c715af")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<person_list>)))
  "Returns full string definition for message of type '<person_list>"
  (cl:format cl:nil "vision/person[] list~%================================================================================~%MSG: vision/person~%string name~%int64 x~%int64 y~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'person_list)))
  "Returns full string definition for message of type 'person_list"
  (cl:format cl:nil "vision/person[] list~%================================================================================~%MSG: vision/person~%string name~%int64 x~%int64 y~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <person_list>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'list) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <person_list>))
  "Converts a ROS message object to a list"
  (cl:list 'person_list
    (cl:cons ':list (list msg))
))
