#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Geometry>
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection/collision_env.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <srdfdom/srdfdom/model.h>
#include <urdf/model.h>

namespace {
struct Observation {
  bool collision{false}; double distance{std::numeric_limits<double>::quiet_NaN()};
  std::string names[2]; Eigen::Vector3d a{Eigen::Vector3d::Zero()}, b{Eigen::Vector3d::Zero()};
};

moveit_msgs::msg::CollisionObject make_box(double center_x) {
  moveit_msgs::msg::CollisionObject object; object.header.frame_id = "base_link"; object.id = "known_answer_obstacle";
  shape_msgs::msg::SolidPrimitive primitive; primitive.type = shape_msgs::msg::SolidPrimitive::BOX; primitive.dimensions = {0.2, 0.2, 0.2};
  geometry_msgs::msg::Pose pose; pose.position.x = center_x; pose.orientation.w = 1.0;
  object.primitives.push_back(primitive); object.primitive_poses.push_back(pose); object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return object;
}

Observation observe(const moveit::core::RobotModelConstPtr& model, double center_x) {
  planning_scene::PlanningScene scene(model); scene.processCollisionObjectMsg(make_box(center_x));
  scene.allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  const auto& acm = scene.getAllowedCollisionMatrix(); const auto* env = scene.getCollisionEnvUnpadded().get();
  moveit::core::RobotState state(model); state.update();
  collision_detection::CollisionRequest collision_request; collision_request.contacts = true; collision_request.max_contacts = 8;
  collision_request.pad_environment_collisions = false; collision_request.pad_self_collisions = false;
  collision_detection::CollisionResult collision; env->checkRobotCollision(collision_request, collision, state, acm);
  collision_detection::DistanceRequest distance_request; distance_request.type = collision_detection::DistanceRequestTypes::GLOBAL;
  distance_request.enable_signed_distance = true; distance_request.enable_nearest_points = true; distance_request.acm = &acm;
  collision_detection::DistanceResult distance; env->distanceRobot(distance_request, distance, state);
  return {collision.collision, distance.minimum_distance.distance,
          {distance.minimum_distance.link_names[0], distance.minimum_distance.link_names[1]},
          distance.minimum_distance.nearest_points[0], distance.minimum_distance.nearest_points[1]};
}

void check(bool condition, const std::string& message) { if (!condition) throw std::runtime_error(message); }
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: p2b3_c5a_fcl_known_answer URDF SRDF");
    if (!rclcpp::ok()) rclcpp::init(argc, argv);
    auto urdf = std::make_shared<urdf::Model>(); if (!urdf->initFile(argv[1])) throw std::runtime_error("fixture_urdf_parse_failed");
    auto srdf = std::make_shared<srdf::Model>(); if (!srdf->initFile(*urdf, argv[2])) throw std::runtime_error("fixture_srdf_parse_failed");
    auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
    const auto clear = observe(model, 0.4);  // [-0.1,0.1] and [0.3,0.5]: exact 0.2 m gap.
    const auto nearer = observe(model, 0.399);  // exact 0.199 m gap.
    const auto overlap = observe(model, 0.15);  // overlap depth is 0.05 m.
    check(!clear.collision && std::abs(clear.distance - 0.2) < 1e-8, "positive_known_answer_distance_or_free_state_failed");
    const bool points_match = (std::abs(clear.a.x() - 0.1) < 1e-8 && std::abs(clear.b.x() - 0.3) < 1e-8) ||
                              (std::abs(clear.b.x() - 0.1) < 1e-8 && std::abs(clear.a.x() - 0.3) < 1e-8);
    check(points_match, "known_answer_nearest_points_failed:pair=" + clear.names[0] + "|" + clear.names[1] +
                       ":points_x=" + std::to_string(clear.a.x()) + "," + std::to_string(clear.b.x()));
    check(!nearer.collision && std::abs(nearer.distance - 0.199) < 1e-8 && nearer.distance < clear.distance,
          "known_translation_did_not_monotonically_change_distance");
    check(overlap.collision && overlap.distance <= 0.0, "forced_overlap_collision_or_signed_distance_failed");
    std::cout << "{\"status\":\"PASS\",\"clear_distance_m\":" << clear.distance
              << ",\"clear_nearest_points_x_m\":[" << clear.a.x() << ',' << clear.b.x()
              << "],\"pair\":[\"" << clear.names[0] << "\",\"" << clear.names[1] << "\"]"
              << ",\"translated_distance_m\":" << nearer.distance
              << ",\"overlap_distance_m\":" << overlap.distance
              << ",\"overlap_collision\":true}\n";
    rclcpp::shutdown(); return 0;
  } catch (const std::exception& e) {
    std::cerr << "p2b3_c5a_fcl_known_answer: " << e.what() << '\n'; if (rclcpp::ok()) rclcpp::shutdown(); return 2;
  }
}
