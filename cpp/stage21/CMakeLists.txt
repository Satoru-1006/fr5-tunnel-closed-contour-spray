cmake_minimum_required(VERSION 3.22)
project(stage21_native_diagnostics)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(rclcpp REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(urdf REQUIRED)

add_executable(stage21_native_contact_probe stage21_native_contact_probe.cpp)
target_compile_features(stage21_native_contact_probe PUBLIC cxx_std_17)
ament_target_dependencies(stage21_native_contact_probe
  geometry_msgs moveit_core moveit_msgs moveit_ros_planning rclcpp shape_msgs urdf)
target_link_libraries(stage21_native_contact_probe
  moveit_core::moveit_collision_detection_bullet
  moveit_core::moveit_collision_detection_fcl)

install(TARGETS stage21_native_contact_probe DESTINATION lib/${PROJECT_NAME})
ament_package()
