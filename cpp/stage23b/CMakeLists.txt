cmake_minimum_required(VERSION 3.16)
project(stage23b_bullet_representation_overlay)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FCL REQUIRED fcl)
find_package(Bullet REQUIRED)

add_library(stage23b_bullet_shape_interposer SHARED bullet_shape_interposer.cpp)
ament_target_dependencies(stage23b_bullet_shape_interposer moveit_core)
target_link_libraries(stage23b_bullet_shape_interposer dl)

add_executable(stage23b_formal_probe stage23b_formal_probe.cpp)
ament_target_dependencies(stage23b_formal_probe geometry_msgs moveit_core moveit_ros_planning moveit_msgs rclcpp shape_msgs)
target_link_libraries(stage23b_formal_probe Eigen3::Eigen dl)

add_executable(stage23b_geometry_probe stage23b_geometry_probe.cpp)
target_include_directories(stage23b_geometry_probe PRIVATE ${FCL_INCLUDE_DIRS} ${BULLET_INCLUDE_DIRS})
target_link_directories(stage23b_geometry_probe PRIVATE ${FCL_LIBRARY_DIRS} ${BULLET_LIBRARY_DIRS})
target_link_libraries(stage23b_geometry_probe PRIVATE ${FCL_LIBRARIES} BulletCollision BulletDynamics LinearMath)

install(TARGETS stage23b_bullet_shape_interposer
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)
install(TARGETS stage23b_formal_probe DESTINATION lib/${PROJECT_NAME})
install(TARGETS stage23b_geometry_probe DESTINATION lib/${PROJECT_NAME})

ament_package()
