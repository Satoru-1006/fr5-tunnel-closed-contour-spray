cmake_minimum_required(VERSION 3.16)
project(stage24_closed_loop_graph)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(stage24_formal_probe stage24_formal_probe.cpp)
ament_target_dependencies(stage24_formal_probe geometry_msgs moveit_core moveit_ros_planning moveit_msgs rclcpp shape_msgs)
target_link_libraries(stage24_formal_probe Eigen3::Eigen dl)

install(TARGETS stage24_formal_probe DESTINATION lib/${PROJECT_NAME})
ament_package()
