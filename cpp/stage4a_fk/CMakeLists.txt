cmake_minimum_required(VERSION 3.22)
project(stage4a_fk)

find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(stage4a_fk stage4a_fk.cpp)
target_compile_features(stage4a_fk PRIVATE cxx_std_17)
ament_target_dependencies(stage4a_fk moveit_core rclcpp)

install(TARGETS stage4a_fk DESTINATION lib/${PROJECT_NAME})
ament_package()
