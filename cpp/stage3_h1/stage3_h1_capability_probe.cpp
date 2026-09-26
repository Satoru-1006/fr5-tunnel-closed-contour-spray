// Stage 3 H1 offline-only MoveIt2 capability qualification.
//
// This probe constructs a synthetic PlanningScene, calls the installed
// CollisionEnv APIs, and writes evidence. It never creates an action client,
// sends a FollowJointTrajectory goal, or connects to a robot/controller.
#define main stage3_h1_embedded_stage23a7_main
#include "../stage23a7/stage23a7_runtime_audit_probe.cpp"
#undef main

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection/collision_matrix.hpp>
#include <moveit/collision_detection_bullet/collision_detector_allocator_bullet.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/link_model.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

namespace fs = std::filesystem;
using moveit::core::RobotState;

static void json_string(std::ostream& out, const std::string& value)
{
  out << '"';
  for (const char c : value) {
    if (c == '"' || c == '\\')
      out << '\\';
    if (c == '\n')
      out << "\\n";
    else if (c == '\r')
      out << "\\r";
    else
      out << c;
  }
  out << '"';
}

static std::string number(double value)
{
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

static moveit_msgs::msg::CollisionObject box_object(const std::string& id, const Eigen::Vector3d& center, double side)
{
  moveit_msgs::msg::CollisionObject object;
  object.header.frame_id = "base_link";
  object.id = id;
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  primitive.dimensions = { side, side, side };
  geometry_msgs::msg::Pose pose;
  pose.orientation.w = 1.0;
  pose.position.x = center.x();
  pose.position.y = center.y();
  pose.position.z = center.z();
  object.primitives.push_back(primitive);
  object.primitive_poses.push_back(pose);
  object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return object;
}

static collision_detection::CollisionRequest request_for(const std::string& group)
{
  collision_detection::CollisionRequest request;
  request.group_name = group;
  request.contacts = true;
  request.max_contacts = 8;
  request.max_contacts_per_pair = 1;
  request.pad_environment_collisions = true;
  request.pad_self_collisions = false;
  return request;
}

static collision_detection::DistanceRequest distance_request_for(const std::string& group,
                                                                 const collision_detection::AllowedCollisionMatrix& acm)
{
  collision_detection::DistanceRequest request;
  request.type = collision_detection::DistanceRequestTypes::GLOBAL;
  request.group_name = group;
  request.enable_nearest_points = true;
  request.enable_signed_distance = true;
  request.acm = &acm;
  return request;
}

static std::unique_ptr<planning_scene::PlanningScene> make_scene(
    const moveit::core::RobotModelConstPtr& model, const Eigen::Vector3d& center, double side,
    const std::string& backend)
{
  auto scene = std::make_unique<planning_scene::PlanningScene>(model);
  scene->processCollisionObjectMsg(box_object("stage3_h1_synthetic_box", center, side));
  if (backend == "bullet")
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create());
  else
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  return scene;
}

static bool finite_distance(const collision_detection::DistanceResult& result)
{
  // MoveIt backends use DBL_MAX as an "unavailable/no pair" sentinel. It is
  // finite in IEEE arithmetic, but it is not a qualified clearance value.
  return std::isfinite(result.minimum_distance.distance) &&
         std::abs(result.minimum_distance.distance) < std::numeric_limits<double>::max() * 0.5;
}

struct DistanceObservation
{
  bool finite = false;
  double value = std::numeric_limits<double>::quiet_NaN();
  bool collision = false;
};

static DistanceObservation observe_robot_world(const moveit::core::RobotModelConstPtr& model,
                                               const std::string& group, const std::string& backend,
                                               const Eigen::Vector3d& center, double side,
                                               const std::vector<double>& positions)
{
  auto scene = make_scene(model, center, side, backend);
  const auto& acm = scene->getAllowedCollisionMatrix();
  const auto request = request_for(group);
  const auto distance_request = distance_request_for(group, acm);
  RobotState state(model);
  state.setVariablePositions(positions);
  state.updateCollisionBodyTransforms();
  collision_detection::CollisionResult collision_result;
  scene->getCollisionEnv()->checkRobotCollision(request, collision_result, state, acm);
  collision_detection::DistanceResult distance_result;
  scene->getCollisionEnv()->distanceRobot(distance_request, distance_result, state);
  return { finite_distance(distance_result), distance_result.minimum_distance.distance, collision_result.collision };
}

static DistanceObservation observe_self(const moveit::core::RobotModelConstPtr& model, const std::string& group,
                                        const std::string& backend, const std::vector<double>& positions)
{
  auto scene = std::make_unique<planning_scene::PlanningScene>(model);
  if (backend == "bullet")
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create());
  else
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  const auto& acm = scene->getAllowedCollisionMatrix();
  const auto request = request_for(group);
  const auto distance_request = distance_request_for(group, acm);
  RobotState state(model);
  state.setVariablePositions(positions);
  state.updateCollisionBodyTransforms();
  collision_detection::CollisionResult collision_result;
  scene->checkSelfCollision(request, collision_result, state, acm);
  collision_detection::DistanceResult distance_result;
  scene->getCollisionEnv()->distanceSelf(distance_request, distance_result, state);
  return { finite_distance(distance_result), distance_result.minimum_distance.distance, collision_result.collision };
}

static std::vector<std::pair<std::vector<double>, std::vector<double>>> swept_controls()
{
  std::vector<std::pair<std::vector<double>, std::vector<double>>> controls;
  for (int joint = 0; joint < 6; ++joint) {
    std::vector<double> q0(6, 0.0), q1(6, 0.0);
    q0[joint] = -1.25;
    q1[joint] = 1.25;
    controls.emplace_back(q0, q1);
  }
  for (int joint = 0; joint < 6; ++joint) {
    std::vector<double> q0(6, 0.0), q1(6, 0.0);
    q0[joint] = 1.25;
    q1[joint] = -1.25;
    controls.emplace_back(q0, q1);
  }
  return controls;
}

static bool qualify_bullet_robot_world_ccd(const moveit::core::RobotModelConstPtr& model, const std::string& group,
                                           bool& api_called, bool& endpoint_free, bool& midpoint_collision,
                                           bool& ccd_collision)
{
  api_called = false;
  endpoint_free = false;
  midpoint_collision = false;
  ccd_collision = false;
  const auto links = { "wrist3_link", "wrist2_link", "forearm_link", "upperarm_link", "shoulder_link" };
  for (const auto& pair : swept_controls()) {
    std::vector<double> midpoint(6, 0.0);
    for (size_t j = 0; j < midpoint.size(); ++j)
      midpoint[j] = 0.5 * (pair.first[j] + pair.second[j]);
    RobotState middle(model);
    middle.setVariablePositions(midpoint);
    middle.updateCollisionBodyTransforms();
    for (const auto& link_name : links) {
      const auto* link = model->getLinkModel(link_name);
      if (!link || link->getShapes().empty())
        continue;
      const Eigen::Vector3d center = middle.getGlobalLinkTransform(link).translation();
      for (const double side : { 0.015, 0.025, 0.04, 0.06, 0.10 }) {
        auto scene = make_scene(model, center, side, "bullet");
        const auto& acm = scene->getAllowedCollisionMatrix();
        const auto request = request_for(group);
        RobotState start(model), end(model);
        start.setVariablePositions(pair.first);
        end.setVariablePositions(pair.second);
        start.updateCollisionBodyTransforms();
        end.updateCollisionBodyTransforms();
        collision_detection::CollisionResult start_result, end_result, middle_result;
        scene->getCollisionEnv()->checkRobotCollision(request, start_result, start, acm);
        scene->getCollisionEnv()->checkRobotCollision(request, end_result, end, acm);
        middle_result.clear();
        scene->getCollisionEnv()->checkRobotCollision(request, middle_result, middle, acm);
        if (start_result.collision || end_result.collision || !middle_result.collision)
          continue;
        endpoint_free = true;
        midpoint_collision = true;
        collision_detection::CollisionResult ccd_result;
        scene->getCollisionEnv()->checkRobotCollision(request, ccd_result, start, end, acm);
        api_called = true;
        ccd_collision = ccd_result.collision;
        return true;
      }
    }
  }
  return false;
}

int main(int argc, char** argv)
{
  try {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>(
        "stage3_h1_capability_probe", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
    std::string output_dir;
    std::string group = "fairino5_v6_group";
    node->get_parameter_or("output_dir", output_dir, std::string{});
    node->get_parameter_or("group_name", group, group);
    if (output_dir.empty())
      throw std::runtime_error("output_dir parameter is required");
    fs::create_directories(output_dir);

    robot_model_loader::RobotModelLoader::Options options("robot_description");
    options.load_kinematics_solvers = false;
    robot_model_loader::RobotModelLoader loader(node, options);
    const auto model = loader.getModel();
    if (!model)
      throw std::runtime_error("MoveIt RobotModelLoader returned no model");
    const auto* reference_link = model->getLinkModel("wrist3_link");
    if (!reference_link)
      throw std::runtime_error("wrist3_link is missing from the loaded MoveIt model");
    std::vector<double> q0(6, 0.0);
    RobotState reference(model);
    reference.setVariablePositions(q0);
    reference.updateCollisionBodyTransforms();
    const Eigen::Vector3d reference_center = reference.getGlobalLinkTransform(reference_link).translation();

    std::ofstream records(fs::path(output_dir) / "stage3_h1_capability_probe_records.jsonl");
    std::ofstream report(fs::path(output_dir) / "stage3_h1_capability_probe.json");
    report << "{\"schema_version\":\"stage3-h1-capability-probe-v1\","
           << "\"offline_only\":true,\"fjt_goal_sent\":false,\"robot_motion_started\":false,"
           << "\"robot_model\":\"" << model->getName() << "\",\"group_name\":\"" << group << "\","
           << "\"reference_link\":\"wrist3_link\",\"backends\":{";

    const std::vector<std::string> backends = { "fcl", "bullet" };
    for (size_t backend_index = 0; backend_index < backends.size(); ++backend_index) {
      const std::string& backend = backends[backend_index];
      const Eigen::Vector3d far_center = reference_center + Eigen::Vector3d(10.0, 10.0, 10.0);
      const Eigen::Vector3d near_center = reference_center + Eigen::Vector3d(0.35, 0.0, 0.0);
      const Eigen::Vector3d contact_center = reference_center;
      const auto far = observe_robot_world(model, group, backend, far_center, 0.05, q0);
      const auto near = observe_robot_world(model, group, backend, near_center, 0.02, q0);
      const auto contact = observe_robot_world(model, group, backend, contact_center, 0.10, q0);
      const auto self_first = observe_self(model, group, backend, q0);
      const auto self_second = observe_self(model, group, backend, q0);
      const bool world_distance_ordered = far.finite && near.finite && near.value < far.value;
      const bool contact_semantics = contact.collision || (contact.finite && contact.value <= 0.0);
      const bool self_repeatable = self_first.finite && self_second.finite && self_first.value == self_second.value;
      bool api_called = false, endpoint_free = false, midpoint_collision = false, ccd_collision = false;
      const bool ccd_control_found = backend == "bullet" && qualify_bullet_robot_world_ccd(
          model, group, api_called, endpoint_free, midpoint_collision, ccd_collision);

      records << "{\"backend\":\"" << backend << "\",\"far_distance_m\":" << number(far.value)
              << ",\"near_distance_m\":" << number(near.value) << ",\"contact_distance_m\":"
              << number(contact.value) << ",\"far_collision\":" << (far.collision ? "true" : "false")
              << ",\"near_collision\":" << (near.collision ? "true" : "false")
              << ",\"contact_collision\":" << (contact.collision ? "true" : "false")
              << ",\"world_distance_ordered\":" << (world_distance_ordered ? "true" : "false")
              << ",\"contact_semantics\":" << (contact_semantics ? "true" : "false")
              << ",\"self_distance_m\":" << number(self_first.value)
              << ",\"self_distance_repeat_m\":" << number(self_second.value)
              << ",\"self_repeatable\":" << (self_repeatable ? "true" : "false")
              << ",\"two_state_api_called\":" << (api_called ? "true" : "false")
              << ",\"ccd_control_found\":" << (ccd_control_found ? "true" : "false")
              << ",\"endpoint_free\":" << (endpoint_free ? "true" : "false")
              << ",\"midpoint_collision\":" << (midpoint_collision ? "true" : "false")
              << ",\"ccd_collision\":" << (ccd_collision ? "true" : "false") << "}\n";

      if (backend_index)
        report << ',';
      json_string(report, backend);
      report << ":{\"far_distance_m\":" << number(far.value) << ",\"near_distance_m\":" << number(near.value)
             << ",\"contact_distance_m\":" << number(contact.value) << ",\"far_collision\":"
             << (far.collision ? "true" : "false") << ",\"near_collision\":" << (near.collision ? "true" : "false")
             << ",\"contact_collision\":" << (contact.collision ? "true" : "false")
             << ",\"distance_api_verified\":" << (world_distance_ordered && contact_semantics ? "true" : "false")
             << ",\"self_distance_api_verified\":" << (self_repeatable ? "true" : "false")
             << ",\"two_state_api_called\":" << (api_called ? "true" : "false")
             << ",\"ccd_control_found\":" << (ccd_control_found ? "true" : "false")
             << ",\"endpoint_free\":" << (endpoint_free ? "true" : "false")
             << ",\"midpoint_collision\":" << (midpoint_collision ? "true" : "false")
             << ",\"ccd_collision\":" << (ccd_collision ? "true" : "false") << '}';
    }
    report << "},\"continuous_self_collision_api\":\"not_present_in_collision_env_interface\","
           << "\"distance_signed_semantics_requested\":true,\"deterministic_fixture\":true}\n";
    report.close();
    records.close();
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "stage3_h1_capability_probe fatal: " << error.what() << '\n';
    if (rclcpp::ok())
      rclcpp::shutdown();
    return 20;
  }
}
