cmake_minimum_required(VERSION 3.16)
project(p2b3_c5a_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(shape_msgs REQUIRED)

add_executable(p2b3_c5a_native p2b3_c5a_native.cpp)
ament_target_dependencies(p2b3_c5a_native geometry_msgs moveit_core moveit_msgs rclcpp shape_msgs)
target_link_libraries(p2b3_c5a_native Eigen3::Eigen)
target_link_libraries(p2b3_c5a_native moveit_core::moveit_collision_detection_fcl)

add_executable(p2b3_c5a_fcl_known_answer p2b3_c5a_fcl_known_answer.cpp)
ament_target_dependencies(p2b3_c5a_fcl_known_answer geometry_msgs moveit_core moveit_msgs rclcpp shape_msgs)
target_link_libraries(p2b3_c5a_fcl_known_answer Eigen3::Eigen)
target_link_libraries(p2b3_c5a_fcl_known_answer moveit_core::moveit_collision_detection_fcl)

install(TARGETS p2b3_c5a_native p2b3_c5a_fcl_known_answer DESTINATION lib/${PROJECT_NAME})
ament_package()
