; Auto-generated. Do not edit!


(cl:in-package vision-msg)


;//! \htmlinclude img_list.msg.html

(cl:defclass <img_list> (roslisp-msg-protocol:ros-message)
  ((images
    :reader images
    :initarg :images
    :type (cl:vector vision-msg:img)
   :initform (cl:make-array 0 :element-type 'vision-msg:img :initial-element (cl:make-instance 'vision-msg:img))))
)

(cl:defclass img_list (<img_list>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <img_list>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'img_list)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name vision-msg:<img_list> is deprecated: use vision-msg:img_list instead.")))

(cl:ensure-generic-function 'images-val :lambda-list '(m))
(cl:defmethod images-val ((m <img_list>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader vision-msg:images-val is deprecated.  Use vision-msg:images instead.")
  (images m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <img_list>) ostream)
  "Serializes a message object of type '<img_list>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'images))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'images))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <img_list>) istream)
  "Deserializes a message object of type '<img_list>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'images) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'images)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'vision-msg:img))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<img_list>)))
  "Returns string type for a message object of type '<img_list>"
  "vision/img_list")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'img_list)))
  "Returns string type for a message object of type 'img_list"
  "vision/img_list")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<img_list>)))
  "Returns md5sum for a message object of type '<img_list>"
  "beeae6a30f35d07ad6b429e4f70ceea5")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'img_list)))
  "Returns md5sum for a message object of type 'img_list"
  "beeae6a30f35d07ad6b429e4f70ceea5")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<img_list>)))
  "Returns full string definition for message of type '<img_list>"
  (cl:format cl:nil "vision/img[] images~%================================================================================~%MSG: vision/img~%int64 x~%int64 y~%int64 w~%int64 h~%string name~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'img_list)))
  "Returns full string definition for message of type 'img_list"
  (cl:format cl:nil "vision/img[] images~%================================================================================~%MSG: vision/img~%int64 x~%int64 y~%int64 w~%int64 h~%string name~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <img_list>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'images) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <img_list>))
  "Converts a ROS message object to a list"
  (cl:list 'img_list
    (cl:cons ':images (images msg))
))
