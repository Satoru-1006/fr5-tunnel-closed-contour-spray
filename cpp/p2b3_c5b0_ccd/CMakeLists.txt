cmake_minimum_required(VERSION 3.16)
project(p2b3_c5b0_ccd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Bullet REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(rclcpp REQUIRED)
find_package(shape_msgs REQUIRED)

add_executable(p2b3_c5b0_bullet_fixture p2b3_c5b0_bullet_fixture.cpp)
target_include_directories(p2b3_c5b0_bullet_fixture PRIVATE ${BULLET_INCLUDE_DIRS})
target_link_libraries(p2b3_c5b0_bullet_fixture PRIVATE ${BULLET_LIBRARIES})

add_executable(p2b3_c5b0_fr5_segment_ccd p2b3_c5b0_fr5_segment_ccd.cpp)
ament_target_dependencies(p2b3_c5b0_fr5_segment_ccd geometry_msgs moveit_core moveit_msgs moveit_ros_planning rclcpp shape_msgs)
target_link_libraries(p2b3_c5b0_fr5_segment_ccd Eigen3::Eigen)

install(TARGETS p2b3_c5b0_bullet_fixture p2b3_c5b0_fr5_segment_ccd DESTINATION lib/${PROJECT_NAME})
ament_package()
