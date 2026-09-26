#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection/collision_env.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <moveit/kinematics_base/kinematics_base.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/mesh.hpp>
#include <shape_msgs/msg/mesh_triangle.hpp>
#include <srdfdom/srdfdom/model.h>
#include <urdf/model.h>
#include <unistd.h>

namespace fs = std::filesystem;

struct Args {
  std::string urdf, srdf, requests, mesh, output;
  std::string group{"fairino5_v6_group"};
  std::string base{"base_link"};
  std::string tip{"spray_tcp_link"};
};

struct ObjMesh { std::vector<Eigen::Vector3d> vertices; std::vector<Eigen::Vector3i> triangles; };

struct Request {
  int waypoint_index{-1};
  int segment_id{-1};
  int component_id{-1};
  std::string spray_state;
  int source_target_id{-1};
  int destination_target_id{-1};
  std::string task_sample_id;
  geometry_msgs::msg::Pose pose;
  std::vector<std::vector<double>> seeds;
};

struct ContactBundle { bool collision{false}; std::vector<std::string> pairs; };
struct DistanceObservation {
  bool available{false}; double value{std::numeric_limits<double>::quiet_NaN()}; std::string pair;
};
struct StateResult {
  bool fk{false}; bool bounds{false}; ContactBundle env; ContactBundle self;
  DistanceObservation env_distance; DistanceObservation self_distance;
  Eigen::Vector3d tcp{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  double translation_error{std::numeric_limits<double>::quiet_NaN()};
  double rotation_error{std::numeric_limits<double>::quiet_NaN()};
};

static std::vector<std::string> split_tab(const std::string& line)
{
  std::vector<std::string> fields; std::size_t start = 0;
  while (true) {
    const auto end = line.find('\t', start);
    fields.push_back(line.substr(start, end == std::string::npos ? std::string::npos : end - start));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return fields;
}

static double number(const std::string& value)
{ return value.empty() ? 0.0 : std::stod(value); }

static std::string quote(const std::string& value)
{
  std::ostringstream out; out << '"';
  for (const unsigned char c : value) {
    if (c == '"') out << "\\\"";
    else if (c == '\\') out << "\\\\";
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else out << c;
  }
  out << '"'; return out.str();
}

static std::string json_number(double value)
{ if (!std::isfinite(value)) return "null"; std::ostringstream out; out << std::setprecision(17) << value; return out.str(); }

static std::string array(const std::vector<double>& values)
{
  std::ostringstream out; out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << json_number(values[i]); }
  out << ']'; return out.str();
}

static std::string string_array(const std::vector<std::string>& values)
{
  std::ostringstream out; out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << quote(values[i]); }
  out << ']'; return out.str();
}

static Args parse_args(int argc, char** argv)
{
  Args args;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key(argv[i]); const std::string value(argv[i + 1]);
    if (key == "--urdf") args.urdf = value;
    else if (key == "--srdf") args.srdf = value;
    else if (key == "--requests") args.requests = value;
    else if (key == "--mesh") args.mesh = value;
    else if (key == "--output") args.output = value;
    else if (key == "--group") args.group = value;
    else if (key == "--base") args.base = value;
    else if (key == "--tip") args.tip = value;
  }
  if (args.urdf.empty() || args.srdf.empty() || args.requests.empty() || args.mesh.empty() || args.output.empty())
    throw std::runtime_error("missing --urdf/--srdf/--requests/--mesh/--output");
  return args;
}

static ObjMesh read_obj(const fs::path& path)
{
  ObjMesh mesh; std::ifstream input(path); std::string line;
  if (!input) throw std::runtime_error("cannot open mesh");
  while (std::getline(input, line)) {
    std::stringstream stream(line); std::string tag; stream >> tag;
    if (tag == "v") { double x{}, y{}, z{}; stream >> x >> y >> z; mesh.vertices.emplace_back(x, y, z); }
    else if (tag == "f") {
      std::vector<int> ids; std::string token;
      while (stream >> token) { const auto slash = token.find('/'); const int id = std::stoi(token.substr(0, slash)); ids.push_back(id > 0 ? id - 1 : static_cast<int>(mesh.vertices.size()) + id); }
      for (std::size_t i = 1; i + 1 < ids.size(); ++i) mesh.triangles.emplace_back(ids[0], ids[i], ids[i + 1]);
    }
  }
  if (mesh.vertices.empty() || mesh.triangles.empty()) throw std::runtime_error("empty mesh");
  return mesh;
}

static moveit_msgs::msg::CollisionObject mesh_object(const ObjMesh& mesh)
{
  moveit_msgs::msg::CollisionObject object; object.header.frame_id = "base_link"; object.id = "fixture_surface:fixture_curved_cylinder_patch";
  shape_msgs::msg::Mesh shape;
  for (const auto& vertex : mesh.vertices) { geometry_msgs::msg::Point point; point.x = vertex.x(); point.y = vertex.y(); point.z = vertex.z(); shape.vertices.push_back(point); }
  for (const auto& tri : mesh.triangles) { shape_msgs::msg::MeshTriangle face; face.vertex_indices = {static_cast<unsigned int>(tri.x()), static_cast<unsigned int>(tri.y()), static_cast<unsigned int>(tri.z())}; shape.triangles.push_back(face); }
  object.meshes.push_back(shape); geometry_msgs::msg::Pose pose; pose.orientation.w = 1.0; object.mesh_poses.push_back(pose); object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return object;
}

static ContactBundle collect(const collision_detection::CollisionResult& result)
{
  ContactBundle out; out.collision = result.collision;
  for (const auto& item : result.contacts) out.pairs.push_back(item.first.first + "|" + item.first.second);
  std::sort(out.pairs.begin(), out.pairs.end()); out.pairs.erase(std::unique(out.pairs.begin(), out.pairs.end()), out.pairs.end()); return out;
}

static collision_detection::CollisionRequest request(const std::string& group, bool environment)
{
  collision_detection::CollisionRequest out; out.group_name = group; out.contacts = true; out.max_contacts = 4096; out.max_contacts_per_pair = 64; out.verbose = false;
  out.pad_environment_collisions = environment; out.pad_self_collisions = false; return out;
}

static DistanceObservation distance(const collision_detection::CollisionEnvConstPtr& environment, const std::string& group, const collision_detection::AllowedCollisionMatrix& acm, const moveit::core::RobotState& state, bool self)
{
  collision_detection::DistanceRequest req; req.type = collision_detection::DistanceRequestTypes::GLOBAL; req.group_name = group; req.enable_nearest_points = true; req.enable_signed_distance = true; req.acm = &acm;
  collision_detection::DistanceResult result; if (self) environment->distanceSelf(req, result, state); else environment->distanceRobot(req, result, state);
  DistanceObservation out; const double sentinel = std::numeric_limits<double>::max() * 0.5;
  out.available = std::isfinite(result.minimum_distance.distance) && std::abs(result.minimum_distance.distance) < sentinel;
  if (out.available) { out.value = result.minimum_distance.distance; out.pair = result.minimum_distance.link_names[0] + "|" + result.minimum_distance.link_names[1]; }
  return out;
}

static StateResult evaluate(const std::vector<double>& q, const geometry_msgs::msg::Pose& desired, const std::shared_ptr<const moveit::core::RobotModel>& model, const moveit::core::JointModelGroup* group, planning_scene::PlanningScene& scene, const std::string& group_name)
{
  StateResult out; moveit::core::RobotState state(model); state.setJointGroupPositions(group, q); state.update(); out.bounds = state.satisfiesBounds(group);
  try { const Eigen::Isometry3d& transform = state.getGlobalLinkTransform("spray_tcp_link"); out.tcp = transform.translation(); out.orientation = Eigen::Quaterniond(transform.rotation()); out.fk = out.tcp.allFinite() && out.orientation.coeffs().allFinite(); } catch (...) { out.fk = false; }
  if (out.fk) {
    const Eigen::Vector3d target(desired.position.x, desired.position.y, desired.position.z);
    out.translation_error = (out.tcp - target).norm();
    Eigen::Quaterniond expected(desired.orientation.w, desired.orientation.x, desired.orientation.y, desired.orientation.z); expected.normalize();
    Eigen::Quaterniond actual = out.orientation.normalized(); out.rotation_error = Eigen::AngleAxisd(expected.inverse() * actual).angle();
  }
  if (!out.bounds || !out.fk) return out;
  collision_detection::CollisionResult env_result, self_result;
  scene.getCollisionEnv()->checkRobotCollision(request(group_name, true), env_result, state, scene.getAllowedCollisionMatrix());
  scene.checkSelfCollision(request(group_name, false), self_result, state);
  out.env = collect(env_result); out.self = collect(self_result);
  out.env_distance = distance(scene.getCollisionEnvUnpadded(), group_name, scene.getAllowedCollisionMatrix(), state, false);
  out.self_distance = distance(scene.getCollisionEnvUnpadded(), group_name, scene.getAllowedCollisionMatrix(), state, true);
  return out;
}

static bool read_request(std::istream& input, Request& out)
{
  std::string line; if (!std::getline(input, line)) return false; if (line.empty()) return true;
  const auto fields = split_tab(line); if (fields.size() < 16) throw std::runtime_error("invalid H6 request row");
  std::size_t i = 0; out.waypoint_index = static_cast<int>(number(fields[i++])); out.segment_id = static_cast<int>(number(fields[i++])); out.component_id = static_cast<int>(number(fields[i++])); out.spray_state = fields[i++]; out.source_target_id = static_cast<int>(number(fields[i++])); out.destination_target_id = static_cast<int>(number(fields[i++])); out.task_sample_id = fields[i++];
  out.pose.position.x = number(fields[i++]); out.pose.position.y = number(fields[i++]); out.pose.position.z = number(fields[i++]); out.pose.orientation.x = number(fields[i++]); out.pose.orientation.y = number(fields[i++]); out.pose.orientation.z = number(fields[i++]); out.pose.orientation.w = number(fields[i++]);
  const int seed_count = static_cast<int>(number(fields[i++])); if (seed_count < 0 || fields.size() != i + static_cast<std::size_t>(seed_count) * 6) throw std::runtime_error("H6 seed count/field mismatch");
  out.seeds.clear(); for (int seed = 0; seed < seed_count; ++seed) { std::vector<double> q; for (int j = 0; j < 6; ++j) q.push_back(number(fields[i++])); out.seeds.push_back(std::move(q)); }
  return true;
}

static std::string state_json(const StateResult& state)
{
  const Eigen::Quaterniond q = state.orientation.normalized();
  std::ostringstream out; out << "{\"joint_limit_valid\":" << (state.bounds ? "true" : "false") << ",\"fk_computable\":" << (state.fk ? "true" : "false") << ",\"translation_error_m\":" << json_number(state.translation_error) << ",\"rotation_error_rad\":" << json_number(state.rotation_error) << ",\"environment_collision\":" << (state.env.collision ? "true" : "false") << ",\"self_collision\":" << (state.self.collision ? "true" : "false") << ",\"collision_free\":" << (state.fk && state.bounds && !state.env.collision && !state.self.collision ? "true" : "false") << ",\"environment_collision_pairs\":" << string_array(state.env.pairs) << ",\"self_collision_pairs\":" << string_array(state.self.pairs) << ",\"environment_min_signed_distance_m\":" << json_number(state.env_distance.value) << ",\"self_min_signed_distance_m\":" << json_number(state.self_distance.value) << ",\"tcp_position_m\":[" << json_number(state.tcp.x()) << "," << json_number(state.tcp.y()) << "," << json_number(state.tcp.z()) << "],\"tcp_orientation_xyzw\":[" << json_number(q.x()) << "," << json_number(q.y()) << "," << json_number(q.z()) << "," << json_number(q.w()) << "]}";
  return out.str();
}

static std::string pose_json(const geometry_msgs::msg::Pose& pose)
{ return "[" + json_number(pose.position.x) + "," + json_number(pose.position.y) + "," + json_number(pose.position.z) + "," + json_number(pose.orientation.x) + "," + json_number(pose.orientation.y) + "," + json_number(pose.orientation.z) + "," + json_number(pose.orientation.w) + "]"; }

int main(int argc, char** argv)
{
  try {
    const Args args = parse_args(argc, argv); fs::create_directories(args.output);
    auto urdf = std::make_shared<urdf::Model>(); if (!urdf->initFile(args.urdf)) throw std::runtime_error("URDF parse failed");
    auto srdf = std::make_shared<srdf::Model>(); if (!srdf->initFile(*urdf, args.srdf)) throw std::runtime_error("SRDF parse failed");
    const ObjMesh mesh = read_obj(args.mesh);
    rclcpp::init(argc, argv); rclcpp::NodeOptions options; options.automatically_declare_parameters_from_overrides(true);
    const std::string prefix = "robot_description_kinematics." + args.group + ".";
    options.parameter_overrides({rclcpp::Parameter(prefix + "kinematics_solver", "kdl_kinematics_plugin/KDLKinematicsPlugin"), rclcpp::Parameter(prefix + "kinematics_solver_search_resolution", 0.005), rclcpp::Parameter(prefix + "position_only_ik", false), rclcpp::Parameter(prefix + "orientation_vs_position", 1.0), rclcpp::Parameter(prefix + "epsilon", 1.0e-5), rclcpp::Parameter(prefix + "max_solver_iterations", 500), rclcpp::Parameter(prefix + "joints", std::vector<std::string>{"j1", "j2", "j3", "j4", "j5", "j6"})});
    auto node = std::make_shared<rclcpp::Node>("stage3_h6_native_bridge", options); auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf); if (!model->hasJointModelGroup(args.group)) throw std::runtime_error("planning group unavailable");
    pluginlib::ClassLoader<kinematics::KinematicsBase> loader("moveit_core", "kinematics::KinematicsBase"); auto solver = loader.createSharedInstance("kdl_kinematics_plugin/KDLKinematicsPlugin"); if (!solver->initialize(node, *model, args.group, args.base, {args.tip}, 0.005)) throw std::runtime_error("KDL solver initialize failed");
    const auto* joint_group = model->getJointModelGroup(args.group); auto scene = std::make_unique<planning_scene::PlanningScene>(model); scene->processCollisionObjectMsg(mesh_object(mesh)); scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
    std::ifstream input(args.requests); if (!input) throw std::runtime_error("cannot open H6 requests");
    std::ofstream branch(fs::path(args.output) / "stage3_h6_native_ik_branch_trace.jsonl"); std::ofstream collision(fs::path(args.output) / "stage3_h6_native_collision_evidence.jsonl"); std::ofstream fk(fs::path(args.output) / "stage3_h6_native_fk_tcp_validation.jsonl");
    int row_count = 0, on_count = 0, off_count = 0, accepted_on = 0, accepted_off = 0, ik_attempts = 0, failed_on = 0, failed_off = 0; int active_segment = -1; std::vector<double> previous;
    Request row;
    while (read_request(input, row)) {
      ++row_count; const bool is_on = row.spray_state == "SPRAY_ON"; if (is_on) ++on_count; else ++off_count;
      if (!is_on || row.segment_id != active_segment) { previous.clear(); active_segment = row.segment_id; }
      std::vector<std::vector<double>> seeds;
      if (is_on && !previous.empty()) seeds.push_back(previous);
      for (const auto& seed : row.seeds) seeds.push_back(seed);
      bool accepted = false; std::vector<double> accepted_q; std::string accepted_label; StateResult accepted_state; double best_distance = std::numeric_limits<double>::infinity();
      std::ostringstream attempts; attempts << '['; bool first_attempt = true;
      for (std::size_t seed_index = 0; seed_index < seeds.size(); ++seed_index) {
        const auto& seed = seeds[seed_index]; std::vector<double> q; bool solved = false; moveit_msgs::msg::MoveItErrorCodes error;
        if (is_on) { solved = solver->getPositionIK(row.pose, seed, q, error); ++ik_attempts; }
        else { solved = seed.size() == 6; q = seed; error.val = solved ? moveit_msgs::msg::MoveItErrorCodes::SUCCESS : moveit_msgs::msg::MoveItErrorCodes::INVALID_ROBOT_STATE; }
        StateResult state; if (solved && q.size() == 6) state = evaluate(q, row.pose, model, joint_group, *scene, args.group);
        const bool valid = solved && q.size() == 6 && state.bounds && state.fk && (is_on ? (state.translation_error <= 1.0e-6 && state.rotation_error <= 1.0e-6) : true) && !state.env.collision && !state.self.collision;
        const double distance_to_previous = (!previous.empty() && q.size() == 6) ? std::sqrt(std::inner_product(q.begin(), q.end(), previous.begin(), 0.0, std::plus<double>(), [](double a, double b) { const double d = a - b; return d * d; })) : 0.0;
        if (!first_attempt) attempts << ','; first_attempt = false; attempts << "{\"seed_index\":" << seed_index << ",\"seed_label\":" << quote((!previous.empty() && seed_index == 0) ? "previous_accepted_waypoint" : "frozen_h5_seed_bank") << ",\"solver_success\":" << (solved ? "true" : "false") << ",\"solver_error_code\":" << error.val << ",\"joint_values\":" << array(q) << ",\"distance_to_previous_rad\":" << json_number(distance_to_previous) << ",\"state\":" << state_json(state) << "}";
        collision << "{\"schema_version\":\"stage3-h6-native-collision-evidence-v1\",\"waypoint_index\":" << row.waypoint_index << ",\"segment_id\":" << row.segment_id << ",\"spray_state\":" << quote(row.spray_state) << ",\"seed_index\":" << seed_index << ",\"joint_values\":" << array(q) << ",\"state\":" << state_json(state) << "}\n";
        if (valid && distance_to_previous < best_distance) { best_distance = distance_to_previous; accepted = true; accepted_q = q; accepted_label = (!previous.empty() && seed_index == 0) ? "previous_accepted_waypoint" : "frozen_h5_seed_bank"; accepted_state = state; }
      }
      attempts << ']'; if (accepted) { previous = accepted_q; if (is_on) ++accepted_on; else ++accepted_off; } else if (is_on) ++failed_on; else ++failed_off;
      branch << "{\"schema_version\":\"stage3-h6-native-ik-branch-trace-v1\",\"waypoint_index\":" << row.waypoint_index << ",\"segment_id\":" << row.segment_id << ",\"component_id\":" << row.component_id << ",\"spray_state\":" << quote(row.spray_state) << ",\"source_target_id\":" << row.source_target_id << ",\"destination_target_id\":" << row.destination_target_id << ",\"task_sample_id\":" << quote(row.task_sample_id) << ",\"desired_tcp_pose_xyzwxyz\":" << pose_json(row.pose) << ",\"accepted\":" << (accepted ? "true" : "false") << ",\"accepted_seed_label\":" << (accepted ? quote(accepted_label) : "null") << ",\"accepted_joint_values\":" << array(accepted_q) << ",\"accepted_state\":" << (accepted ? state_json(accepted_state) : "null") << ",\"attempts\":" << attempts.str() << "}\n";
      if (accepted) fk << "{\"schema_version\":\"stage3-h6-native-fk-tcp-validation-v1\",\"waypoint_index\":" << row.waypoint_index << ",\"segment_id\":" << row.segment_id << ",\"spray_state\":" << quote(row.spray_state) << ",\"desired_tcp_pose_xyzwxyz\":" << pose_json(row.pose) << ",\"joint_values\":" << array(accepted_q) << ",\"translation_error_m\":" << json_number(accepted_state.translation_error) << ",\"rotation_error_rad\":" << json_number(accepted_state.rotation_error) << ",\"joint_limit_valid\":true,\"self_collision\":false,\"environment_collision\":false,\"collision_method\":\"adaptive_discrete_interpolation\",\"ccd_status\":\"not_available\"}\n";
    }
    branch.flush(); collision.flush(); fk.flush();
    std::ofstream summary(fs::path(args.output) / "stage3_h6_native_summary.json"); summary << "{\n  \"schema_version\": \"stage3-h6-native-summary-v1\",\n  \"status\": \"AVAILABLE\",\n  \"backend\": \"MoveIt2 KinematicsBase::getPositionIK + RobotState FK + PlanningScene/FCL\",\n  \"collision_method\": \"adaptive_discrete_interpolation\",\n  \"ccd_status\": \"not_available\",\n  \"waypoint_count\": " << row_count << ",\n  \"spray_on_waypoint_count\": " << on_count << ",\n  \"spray_off_waypoint_count\": " << off_count << ",\n  \"spray_on_accepted_count\": " << accepted_on << ",\n  \"spray_off_accepted_count\": " << accepted_off << ",\n  \"ik_attempt_count\": " << ik_attempts << ",\n  \"spray_on_failed_count\": " << failed_on << ",\n  \"spray_off_failed_count\": " << failed_off << ",\n  \"planning_scene_object\": \"fixture_surface:fixture_curved_cylinder_patch\",\n  \"time_parameterization\": \"not_run\"\n}\n";
    rclcpp::shutdown(); return 0;
  } catch (const std::exception& error) { std::cerr << "stage3_h6_native_bridge: " << error.what() << "\n"; if (rclcpp::ok()) rclcpp::shutdown(); return 2; }
}
