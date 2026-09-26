cmake_minimum_required(VERSION 3.22)
project(stage3_h4_4_native_bridge)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(rclcpp REQUIRED)
find_package(shape_msgs REQUIRED)

add_executable(stage3_h4_4_native_bridge stage3_h4_4_native_bridge.cpp)
target_compile_features(stage3_h4_4_native_bridge PUBLIC cxx_std_17)
ament_target_dependencies(stage3_h4_4_native_bridge
  geometry_msgs moveit_core moveit_msgs moveit_ros_planning rclcpp shape_msgs)
target_link_libraries(stage3_h4_4_native_bridge
  moveit_core::moveit_collision_detection_fcl)

install(TARGETS stage3_h4_4_native_bridge DESTINATION lib/${PROJECT_NAME})
ament_package()
