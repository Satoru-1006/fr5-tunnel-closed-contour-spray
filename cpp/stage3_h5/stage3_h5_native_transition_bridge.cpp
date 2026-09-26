#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection/collision_env.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/mesh.hpp>
#include <shape_msgs/msg/mesh_triangle.hpp>

namespace fs = std::filesystem;

struct ObjMesh { std::vector<Eigen::Vector3d> vertices; std::vector<Eigen::Vector3i> triangles; };

struct Record {
  std::string kind;
  std::string edge_id;
  std::string source_node_id;
  std::string target_node_id;
  int target_index = -1;
  int source_target_index = -1;
  int destination_target_index = -1;
  int state_index = -1;
  int sample_count = 0;
  double alpha = 0.0;
  std::string task_sample_id;
  std::vector<double> q;
};

struct ContactBundle { bool collision = false; std::size_t count = 0; std::vector<std::string> pairs; };
struct DistanceObservation {
  bool available = false;
  bool negative_observed = false;
  double value = std::numeric_limits<double>::quiet_NaN();
  std::string pair;
};

static std::vector<std::string> split_csv(const std::string& line)
{
  std::vector<std::string> fields; std::string field; bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') { field.push_back('"'); ++i; }
      else quoted = !quoted;
    } else if (c == ',' && !quoted) { fields.push_back(field); field.clear(); }
    else field.push_back(c);
  }
  fields.push_back(field);
  if (!fields.empty() && !fields.back().empty() && fields.back().back() == '\r') fields.back().pop_back();
  return fields;
}

static double number(const std::string& value)
{ try { return std::stod(value); } catch (...) { return 0.0; } }

static std::string json_quote(const std::string& value)
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

static std::vector<Record> read_records(const fs::path& path)
{
  std::ifstream input(path); std::string line;
  if (!std::getline(input, line)) return {};
  const auto header = split_csv(line); std::map<std::string, std::size_t> column;
  for (std::size_t i = 0; i < header.size(); ++i) column[header[i]] = i;
  std::vector<Record> result;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split_csv(line);
    auto get = [&](const std::string& name) -> std::string {
      const auto it = column.find(name);
      return it == column.end() || it->second >= fields.size() ? std::string{} : fields[it->second];
    };
    Record row;
    row.kind = get("kind"); row.edge_id = get("edge_id"); row.source_node_id = get("source_node_id");
    row.target_node_id = get("target_node_id"); row.target_index = static_cast<int>(number(get("target_index")));
    row.source_target_index = static_cast<int>(number(get("source_target_index")));
    row.destination_target_index = static_cast<int>(number(get("destination_target_index")));
    row.state_index = static_cast<int>(number(get("state_index")));
    row.sample_count = static_cast<int>(number(get("sample_count"))); row.alpha = number(get("alpha"));
    row.task_sample_id = get("task_sample_id");
    for (int i = 0; i < 6; ++i) row.q.push_back(number(get("q" + std::to_string(i))));
    if (row.q.size() == 6 && !row.kind.empty()) result.push_back(std::move(row));
  }
  return result;
}

static ObjMesh read_obj(const fs::path& path)
{
  ObjMesh mesh; std::ifstream input(path); std::string line;
  while (std::getline(input, line)) {
    std::stringstream stream(line); std::string tag; stream >> tag;
    if (tag == "v") { double x{}, y{}, z{}; stream >> x >> y >> z; mesh.vertices.emplace_back(x, y, z); }
    else if (tag == "f") {
      std::vector<int> indices; std::string token;
      while (stream >> token) { const auto slash = token.find('/'); const int index = std::stoi(token.substr(0, slash)); indices.push_back(index > 0 ? index - 1 : static_cast<int>(mesh.vertices.size()) + index); }
      for (std::size_t i = 1; i + 1 < indices.size(); ++i) mesh.triangles.emplace_back(indices[0], indices[i], indices[i + 1]);
    }
  }
  return mesh;
}

static moveit_msgs::msg::CollisionObject make_mesh_object(const ObjMesh& mesh)
{
  moveit_msgs::msg::CollisionObject object; object.header.frame_id = "base_link"; object.id = "fixture_surface:fixture_curved_cylinder_patch";
  shape_msgs::msg::Mesh shape;
  for (const auto& vertex : mesh.vertices) { geometry_msgs::msg::Point p; p.x = vertex.x(); p.y = vertex.y(); p.z = vertex.z(); shape.vertices.push_back(p); }
  for (const auto& triangle : mesh.triangles) { shape_msgs::msg::MeshTriangle f; f.vertex_indices = {static_cast<unsigned int>(triangle.x()), static_cast<unsigned int>(triangle.y()), static_cast<unsigned int>(triangle.z())}; shape.triangles.push_back(f); }
  object.meshes.push_back(shape); geometry_msgs::msg::Pose pose; pose.orientation.w = 1.0; object.mesh_poses.push_back(pose); object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return object;
}

static ContactBundle collect(const collision_detection::CollisionResult& result)
{
  ContactBundle bundle; bundle.collision = result.collision; bundle.count = result.contact_count;
  for (const auto& entry : result.contacts) bundle.pairs.push_back(entry.first.first + "|" + entry.first.second);
  std::sort(bundle.pairs.begin(), bundle.pairs.end()); bundle.pairs.erase(std::unique(bundle.pairs.begin(), bundle.pairs.end()), bundle.pairs.end());
  return bundle;
}

static collision_detection::CollisionRequest collision_request(const std::string& group, bool pad_environment, bool pad_self)
{
  collision_detection::CollisionRequest request; request.group_name = group; request.contacts = true; request.max_contacts = 4096; request.max_contacts_per_pair = 64;
  request.pad_environment_collisions = pad_environment; request.pad_self_collisions = pad_self; request.verbose = false; return request;
}

static DistanceObservation distance_observation(const collision_detection::CollisionEnvConstPtr& environment, const std::string& group, const collision_detection::AllowedCollisionMatrix& acm, const moveit::core::RobotState& state, bool self)
{
  collision_detection::DistanceRequest request; request.type = collision_detection::DistanceRequestTypes::GLOBAL; request.group_name = group; request.enable_nearest_points = true; request.enable_signed_distance = true; request.acm = &acm;
  collision_detection::DistanceResult result; if (self) environment->distanceSelf(request, result, state); else environment->distanceRobot(request, result, state);
  DistanceObservation observation; const double sentinel = std::numeric_limits<double>::max() * 0.5;
  observation.available = std::isfinite(result.minimum_distance.distance) && std::abs(result.minimum_distance.distance) < sentinel;
  if (observation.available) { observation.value = result.minimum_distance.distance; observation.negative_observed = observation.value < 0.0; observation.pair = result.minimum_distance.link_names[0] + "|" + result.minimum_distance.link_names[1]; }
  return observation;
}

static void write_number_or_null(std::ofstream& out, double value)
{ if (std::isfinite(value)) out << std::setprecision(17) << value; else out << "null"; }

static void write_string_array(std::ofstream& out, const std::vector<std::string>& values)
{ out << '['; for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << json_quote(values[i]); } out << ']'; }

static void write_distance(std::ofstream& out, const DistanceObservation& observation)
{
  out << "{\"status\":" << json_quote(observation.available ? "AVAILABLE" : "NOT_AVAILABLE") << ",\"distance_m\":"; write_number_or_null(out, observation.value);
  out << ",\"negative_distance_observed\":" << (observation.negative_observed ? "true" : "false") << ",\"nearest_pair\":" << (observation.available ? json_quote(observation.pair) : "null") << ",\"signed_distance_request_enabled\":true}";
}

static void write_vec6(std::ofstream& out, const std::vector<double>& q)
{ out << '['; for (std::size_t i = 0; i < q.size(); ++i) { if (i) out << ','; write_number_or_null(out, q[i]); } out << ']'; }

struct StateResult { bool fk = false; bool bounds = false; ContactBundle env; ContactBundle self; DistanceObservation env_distance; DistanceObservation self_distance; Eigen::Vector3d tcp = Eigen::Vector3d::Zero(); Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity(); };

static StateResult evaluate(const Record& row, const std::shared_ptr<const moveit::core::RobotModel>& model, const moveit::core::JointModelGroup* group, planning_scene::PlanningScene& scene, const std::string& group_name)
{
  StateResult result; moveit::core::RobotState state(model); state.setJointGroupPositions(group, row.q); state.update();
  result.bounds = state.satisfiesBounds(group);
  try {
    const Eigen::Isometry3d& transform = state.getGlobalLinkTransform("spray_tcp_link");
    result.tcp = transform.translation(); result.orientation = Eigen::Quaterniond(transform.rotation()); result.fk = result.tcp.allFinite() && result.orientation.coeffs().allFinite();
  } catch (...) { result.fk = false; }
  if (!result.bounds || !result.fk) return result;
  const auto env_request = collision_request(group_name, true, false); const auto self_request = collision_request(group_name, false, false);
  collision_detection::CollisionResult env_result, self_result;
  scene.getCollisionEnv()->checkRobotCollision(env_request, env_result, state, scene.getAllowedCollisionMatrix());
  scene.checkSelfCollision(self_request, self_result, state);
  result.env = collect(env_result); result.self = collect(self_result);
  result.env_distance = distance_observation(scene.getCollisionEnvUnpadded(), group_name, scene.getAllowedCollisionMatrix(), state, false);
  result.self_distance = distance_observation(scene.getCollisionEnvUnpadded(), group_name, scene.getAllowedCollisionMatrix(), state, true);
  return result;
}

static void write_result(std::ofstream& out, const Record& row, const StateResult& result, bool endpoint)
{
  out << "{\"schema_version\":\"stage3-h5-native-state-evidence-v1\",\"kind\":" << json_quote(row.kind) << ",\"endpoint\":" << (endpoint ? "true" : "false")
      << ",\"edge_id\":" << json_quote(row.edge_id) << ",\"source_node_id\":" << json_quote(row.source_node_id) << ",\"target_node_id\":" << json_quote(row.target_node_id)
      << ",\"target_index\":" << row.target_index << ",\"source_target_index\":" << row.source_target_index << ",\"destination_target_index\":" << row.destination_target_index
      << ",\"state_index\":" << row.state_index << ",\"sample_count\":" << row.sample_count << ",\"alpha\":"; write_number_or_null(out, row.alpha);
  out << ",\"joint_values\":"; write_vec6(out, row.q); out << ",\"fk_computable\":" << (result.fk ? "true" : "false") << ",\"joint_limit_valid\":" << (result.bounds ? "true" : "false")
      << ",\"native_fcl_checked\":" << ((result.fk && result.bounds) ? "true" : "false") << ",\"environment_collision\":" << (result.env.collision ? "true" : "false")
      << ",\"self_collision\":" << (result.self.collision ? "true" : "false") << ",\"collision_free\":" << ((result.fk && result.bounds && !result.env.collision && !result.self.collision) ? "true" : "false")
      << ",\"environment_collision_pairs\":"; write_string_array(out, result.env.pairs); out << ",\"self_collision_pairs\":"; write_string_array(out, result.self.pairs);
  out << ",\"collision_environment_objects\":[";
  bool first = true; for (const auto& pair : result.env.pairs) if (pair.find("fixture_surface") != std::string::npos) { if (!first) out << ','; first = false; out << json_quote(pair); } out << ']';
  out << ",\"environment_min_signed_distance\":"; write_distance(out, result.env_distance); out << ",\"self_min_signed_distance\":"; write_distance(out, result.self_distance);
  out << ",\"tcp_position_m\":[" << std::setprecision(17) << result.tcp.x() << ',' << result.tcp.y() << ',' << result.tcp.z() << "]"
      << ",\"tcp_orientation_xyzw\":[" << result.orientation.x() << ',' << result.orientation.y() << ',' << result.orientation.z() << ',' << result.orientation.w() << "]}\n";
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("stage3_h5_native_transition_bridge", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  std::string endpoint_path, transition_path, mesh_path, output_path, group;
  node->get_parameter_or("endpoint_csv", endpoint_path, std::string{}); node->get_parameter_or("transition_csv", transition_path, std::string{}); node->get_parameter_or("curved_mesh_obj", mesh_path, std::string{}); node->get_parameter_or("output_dir", output_path, std::string{}); node->get_parameter_or("group_name", group, std::string("fairino5_v6_group"));
  if (endpoint_path.empty() || transition_path.empty() || mesh_path.empty() || output_path.empty()) { RCLCPP_ERROR(node->get_logger(), "endpoint_csv, transition_csv, curved_mesh_obj and output_dir are required"); rclcpp::shutdown(); return 2; }
  fs::create_directories(output_path); const auto endpoints = read_records(endpoint_path); const auto transitions = read_records(transition_path); const auto mesh = read_obj(mesh_path);
  if (endpoints.empty() || mesh.vertices.empty() || mesh.triangles.empty()) { RCLCPP_ERROR(node->get_logger(), "empty endpoint or mesh input"); rclcpp::shutdown(); return 3; }
  robot_model_loader::RobotModelLoader::Options options("robot_description"); options.load_kinematics_solvers = false; robot_model_loader::RobotModelLoader loader(node, options); const auto model = loader.getModel();
  if (!model) { RCLCPP_ERROR(node->get_logger(), "failed to load robot model"); rclcpp::shutdown(); return 4; }
  const auto* joint_group = model->getJointModelGroup(group); if (!joint_group) { RCLCPP_ERROR(node->get_logger(), "joint group not found: %s", group.c_str()); rclcpp::shutdown(); return 5; }
  auto scene = std::make_unique<planning_scene::PlanningScene>(model); scene->processCollisionObjectMsg(make_mesh_object(mesh)); scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  std::ofstream endpoint_out(fs::path(output_path) / "stage3_h5_endpoint_evidence.jsonl"); std::ofstream transition_out(fs::path(output_path) / "stage3_h5_native_transition_state_evidence.jsonl");
  std::size_t endpoint_count = 0, endpoint_safe = 0, transition_count = 0, transition_collision = 0, transition_fk_failure = 0;
  double min_env = std::numeric_limits<double>::infinity(), min_self = std::numeric_limits<double>::infinity(); std::string min_env_pair, min_self_pair; int min_env_state = -1, min_self_state = -1;
  for (const auto& row : endpoints) { const auto state = evaluate(row, model, joint_group, *scene, group); write_result(endpoint_out, row, state, true); ++endpoint_count; if (state.fk && state.bounds && !state.env.collision && !state.self.collision) ++endpoint_safe; if (state.env_distance.available && state.env_distance.value < min_env) { min_env = state.env_distance.value; min_env_pair = state.env_distance.pair; min_env_state = row.target_index; } if (state.self_distance.available && state.self_distance.value < min_self) { min_self = state.self_distance.value; min_self_pair = state.self_distance.pair; min_self_state = row.target_index; } }
  for (const auto& row : transitions) { const auto state = evaluate(row, model, joint_group, *scene, group); write_result(transition_out, row, state, false); ++transition_count; if (!state.fk || !state.bounds) ++transition_fk_failure; if (state.env.collision || state.self.collision) ++transition_collision; if (state.env_distance.available && state.env_distance.value < min_env) { min_env = state.env_distance.value; min_env_pair = state.env_distance.pair; min_env_state = row.state_index; } if (state.self_distance.available && state.self_distance.value < min_self) { min_self = state.self_distance.value; min_self_pair = state.self_distance.pair; min_self_state = row.state_index; } }
  endpoint_out.flush(); transition_out.flush();
  std::ofstream summary(fs::path(output_path) / "stage3_h5_native_transition_summary.json"); summary << "{\n  \"schema_version\": \"stage3-h5-native-transition-summary-v1\",\n  \"status\": \"AVAILABLE\",\n  \"collision_backend\": \"MoveIt PlanningScene + native FCL\",\n  \"collision_method\": \"adaptive_discrete_interpolation\",\n  \"ccd_status\": \"not_available\",\n  \"endpoint_count\": " << endpoint_count << ",\n  \"endpoint_safe_count\": " << endpoint_safe << ",\n  \"transition_state_count\": " << transition_count << ",\n  \"transition_collision_state_count\": " << transition_collision << ",\n  \"transition_fk_or_bounds_failure_count\": " << transition_fk_failure << ",\n  \"minimum_environment_signed_distance_m\": "; write_number_or_null(summary, min_env); summary << ",\n  \"minimum_environment_state_index\": " << min_env_state << ",\n  \"minimum_environment_nearest_pair\": " << (min_env_pair.empty() ? "null" : json_quote(min_env_pair)) << ",\n  \"minimum_self_signed_distance_m\": "; write_number_or_null(summary, min_self); summary << ",\n  \"minimum_self_state_index\": " << min_self_state << ",\n  \"minimum_self_nearest_pair\": " << (min_self_pair.empty() ? "null" : json_quote(min_self_pair)) << ",\n  \"planning_scene_object\": \"fixture_surface:fixture_curved_cylinder_patch\",\n  \"planning_scene_object_presence\": \"PRESENT\",\n  \"distance_api\": [\"CollisionEnv::distanceRobot\", \"CollisionEnv::distanceSelf\", \"DistanceRequest(enable_signed_distance=true)\"]\n}\n";
  summary.close(); rclcpp::shutdown(); return 0;
}
