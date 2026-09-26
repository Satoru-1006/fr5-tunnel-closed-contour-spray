cmake_minimum_required(VERSION 3.22)
project(stage3_h4_1_bridge)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_kinematics REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(srdfdom REQUIRED)
find_package(urdf REQUIRED)

add_executable(stage3_h4_1_ik_bridge src/stage3_h4_1_ik_bridge.cpp)
target_compile_features(stage3_h4_1_ik_bridge PUBLIC cxx_std_17)
ament_target_dependencies(stage3_h4_1_ik_bridge
  geometry_msgs moveit_core moveit_kinematics moveit_msgs pluginlib rclcpp srdfdom urdf)

install(TARGETS stage3_h4_1_ik_bridge DESTINATION lib/${PROJECT_NAME})
ament_package()
