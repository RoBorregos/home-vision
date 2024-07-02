cmake_minimum_required(VERSION 3.0.2)
project(moondream_action)


find_package(catkin REQUIRED COMPONENTS
  actionlib
  frida_vision_interfaces
  rospy
  cv_bridge
)

#add_action_files(DIRECTORY action FILES moondream.action)
#generate_messages(DEPENDENCIES actionlib_msgs std_msgs)

catkin_package(
	CATKIN_DEPENDS frida_vision_interfaces cv_bridge
	)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


