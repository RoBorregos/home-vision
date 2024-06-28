cmake_minimum_required(VERSION 3.0.2)
project(moondream_action)


find_package(catkin REQUIRED COMPONENTS
  actionlib
  moondream_msgs
  rospy
)

#add_action_files(DIRECTORY action FILES moondream.action)
#generate_messages(DEPENDENCIES actionlib_msgs std_msgs)

catkin_package(
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


