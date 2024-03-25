cmake_minimum_required(VERSION 3.0.2)
project(face_recognition)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

add_message_files(
  FILES
  target_face.msg
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES face_recognition
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
