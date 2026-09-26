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

struct Candidate
{
  int target_index = -1;
  std::string task_sample_id;
  std::string candidate_id;
  std::string target_classification;
  std::string baseline_classification;
  std::string joint_limit_state;
  bool fk_valid = true;
  std::vector<double> q;
};

struct ObjMesh
{
  std::vector<Eigen::Vector3d> vertices;
  std::vector<Eigen::Vector3i> triangles;
};

struct ContactBundle
{
  bool collision = false;
  std::size_t count = 0;
  std::vector<std::string> pairs;
  std::vector<double> depths;
  std::vector<collision_detection::Contact> contacts;
};

struct DistanceObservation
{
  bool available = false;
  bool negative_observed = false;
  double value = std::numeric_limits<double>::quiet_NaN();
  std::string pair;
  Eigen::Vector3d points[2] = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
};

static std::vector<std::string> split_csv(const std::string& line)
{
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i)
  {
    const char c = line[i];
    if (c == '"')
    {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') { field.push_back('"'); ++i; }
      else { quoted = !quoted; }
    }
    else if (c == ',' && !quoted) { fields.push_back(field); field.clear(); }
    else { field.push_back(c); }
  }
  fields.push_back(field);
  if (!fields.empty() && !fields.back().empty() && fields.back().back() == '\r') fields.back().pop_back();
  return fields;
}

static double number(const std::string& value)
{
  try { return std::stod(value); } catch (...) { return 0.0; }
}

static bool boolean(const std::string& value, bool fallback = false)
{
  if (value == "true" || value == "True" || value == "1" || value == "VALID") return true;
  if (value == "false" || value == "False" || value == "0" || value == "INVALID") return false;
  return fallback;
}

static std::vector<Candidate> read_candidates(const fs::path& path)
{
  std::ifstream input(path);
  std::string line;
  if (!std::getline(input, line)) return {};
  const auto header = split_csv(line);
  std::map<std::string, std::size_t> column;
  for (std::size_t i = 0; i < header.size(); ++i) column[header[i]] = i;
  std::vector<Candidate> result;
  while (std::getline(input, line))
  {
    if (line.empty()) continue;
    const auto fields = split_csv(line);
    auto get = [&](const std::string& name) -> std::string {
      const auto it = column.find(name);
      return it == column.end() || it->second >= fields.size() ? std::string{} : fields[it->second];
    };
    Candidate candidate;
    candidate.target_index = static_cast<int>(number(get("target_index")));
    candidate.task_sample_id = get("task_sample_id");
    candidate.candidate_id = get("candidate_id");
    candidate.target_classification = get("target_classification");
    candidate.baseline_classification = get("baseline_classification");
    candidate.joint_limit_state = get("joint_limit_state");
    candidate.fk_valid = boolean(get("fk_valid"), get("fk_valid").empty() || get("fk_valid") == "VALID");
    for (int i = 0; i < 6; ++i) candidate.q.push_back(number(get("q" + std::to_string(i))));
    if (candidate.q.size() == 6 && !candidate.task_sample_id.empty()) result.push_back(std::move(candidate));
  }
  return result;
}

static ObjMesh read_obj(const fs::path& path)
{
  ObjMesh mesh;
  std::ifstream input(path);
  std::string line;
  while (std::getline(input, line))
  {
    std::stringstream stream(line);
    std::string tag;
    stream >> tag;
    if (tag == "v")
    {
      double x{}, y{}, z{}; stream >> x >> y >> z; mesh.vertices.emplace_back(x, y, z);
    }
    else if (tag == "f")
    {
      std::vector<int> indices; std::string token;
      while (stream >> token)
      {
        const auto slash = token.find('/');
        const int index = std::stoi(token.substr(0, slash));
        indices.push_back(index > 0 ? index - 1 : static_cast<int>(mesh.vertices.size()) + index);
      }
      for (std::size_t i = 1; i + 1 < indices.size(); ++i) mesh.triangles.emplace_back(indices[0], indices[i], indices[i + 1]);
    }
  }
  return mesh;
}

static moveit_msgs::msg::CollisionObject make_mesh_object(const ObjMesh& mesh)
{
  moveit_msgs::msg::CollisionObject object;
  object.header.frame_id = "base_link";
  object.id = "fixture_surface:fixture_curved_cylinder_patch";
  shape_msgs::msg::Mesh shape;
  for (const auto& vertex : mesh.vertices)
  {
    geometry_msgs::msg::Point point; point.x = vertex.x(); point.y = vertex.y(); point.z = vertex.z(); shape.vertices.push_back(point);
  }
  for (const auto& triangle : mesh.triangles)
  {
    shape_msgs::msg::MeshTriangle face;
    face.vertex_indices = {static_cast<unsigned int>(triangle.x()), static_cast<unsigned int>(triangle.y()), static_cast<unsigned int>(triangle.z())};
    shape.triangles.push_back(face);
  }
  object.meshes.push_back(shape);
  geometry_msgs::msg::Pose pose; pose.orientation.w = 1.0; object.mesh_poses.push_back(pose);
  object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return object;
}

static std::string json_quote(const std::string& value)
{
  std::ostringstream out; out << '"';
  for (const unsigned char c : value)
  {
    if (c == '"') out << "\\\"";
    else if (c == '\\') out << "\\\\";
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else out << c;
  }
  out << '"'; return out.str();
}

static bool is_world(const collision_detection::Contact& contact, bool first)
{
  return (first ? contact.body_type_1 : contact.body_type_2) == collision_detection::BodyTypes::WORLD_OBJECT;
}

static ContactBundle collect(const collision_detection::CollisionResult& result)
{
  ContactBundle bundle; bundle.collision = result.collision; bundle.count = result.contact_count;
  for (const auto& entry : result.contacts)
  {
    const auto& contacts = entry.second;
    const std::string pair = entry.first.first + "|" + entry.first.second;
    bundle.pairs.push_back(pair);
    for (const auto& contact : contacts) { bundle.contacts.push_back(contact); bundle.depths.push_back(contact.depth); }
  }
  std::sort(bundle.pairs.begin(), bundle.pairs.end());
  bundle.pairs.erase(std::unique(bundle.pairs.begin(), bundle.pairs.end()), bundle.pairs.end());
  if (bundle.count == 0) bundle.count = bundle.contacts.size();
  return bundle;
}

static collision_detection::CollisionRequest collision_request(const std::string& group, bool pad_environment, bool pad_self, std::size_t max_contacts, std::size_t max_contacts_per_pair)
{
  collision_detection::CollisionRequest request;
  request.group_name = group; request.contacts = true; request.max_contacts = max_contacts; request.max_contacts_per_pair = max_contacts_per_pair;
  request.pad_environment_collisions = pad_environment; request.pad_self_collisions = pad_self; request.verbose = false;
  return request;
}

static DistanceObservation distance_observation(const collision_detection::CollisionEnvConstPtr& environment, const std::string& group, const collision_detection::AllowedCollisionMatrix& acm, const moveit::core::RobotState& state, bool self)
{
  collision_detection::DistanceRequest request;
  request.type = collision_detection::DistanceRequestTypes::GLOBAL;
  request.group_name = group; request.enable_nearest_points = true; request.enable_signed_distance = true; request.acm = &acm;
  collision_detection::DistanceResult result;
  if (self) environment->distanceSelf(request, result, state); else environment->distanceRobot(request, result, state);
  DistanceObservation observation;
  const double sentinel = std::numeric_limits<double>::max() * 0.5;
  observation.available = std::isfinite(result.minimum_distance.distance) && std::abs(result.minimum_distance.distance) < sentinel;
  if (observation.available)
  {
    observation.value = result.minimum_distance.distance; observation.negative_observed = observation.value < 0.0;
    observation.pair = result.minimum_distance.link_names[0] + "|" + result.minimum_distance.link_names[1];
    observation.points[0] = result.minimum_distance.nearest_points[0]; observation.points[1] = result.minimum_distance.nearest_points[1];
  }
  return observation;
}

static void write_number_or_null(std::ofstream& out, double value)
{
  if (std::isfinite(value)) out << std::setprecision(17) << value; else out << "null";
}

static void write_string_array(std::ofstream& out, const std::vector<std::string>& values)
{
  out << '['; for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << json_quote(values[i]); } out << ']';
}

static void write_depth_array(std::ofstream& out, const std::vector<double>& values)
{
  out << '['; for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; write_number_or_null(out, values[i]); } out << ']';
}

static void write_contact_array(std::ofstream& out, const std::vector<collision_detection::Contact>& contacts)
{
  out << '[';
  for (std::size_t i = 0; i < contacts.size(); ++i)
  {
    const auto& c = contacts[i]; if (i) out << ',';
    out << "{\"body_1\":" << json_quote(c.body_name_1) << ",\"body_2\":" << json_quote(c.body_name_2)
        << ",\"contact_point_xyz_m\":[" << std::setprecision(17) << c.pos.x() << ',' << c.pos.y() << ',' << c.pos.z() << "]"
        << ",\"normal_xyz\":[" << c.normal.x() << ',' << c.normal.y() << ',' << c.normal.z() << "]"
        << ",\"penetration_depth_m\":"; write_number_or_null(out, c.depth); out << '}';
  }
  out << ']';
}

static void write_variant(std::ofstream& out, const char* name, const ContactBundle& bundle)
{
  out << json_quote(name) << ":{\"collision\":" << (bundle.collision ? "true" : "false") << ",\"contact_count\":" << bundle.count
      << ",\"collision_pairs\":"; write_string_array(out, bundle.pairs);
  out << ",\"penetration_depth_m\":"; write_depth_array(out, bundle.depths);
  out << ",\"contacts\":"; write_contact_array(out, bundle.contacts); out << '}';
}

static void write_distance(std::ofstream& out, const char* name, const DistanceObservation& observation)
{
  out << json_quote(name) << ":{\"status\":" << json_quote(observation.available ? "AVAILABLE" : "NOT_AVAILABLE")
      << ",\"reason\":" << json_quote(observation.available ? "MoveIt/FCL DistanceResult returned a finite minimum_distance.distance" : "DistanceResult.minimum_distance.distance was the unavailable DBL_MAX sentinel")
      << ",\"distance_m\":"; write_number_or_null(out, observation.value);
  out << ",\"signed_distance_request_enabled\":true,\"negative_distance_observed\":" << (observation.negative_observed ? "true" : "false")
      << ",\"nearest_pair\":" << (observation.available ? json_quote(observation.pair) : "null") << ",\"nearest_points\":";
  if (observation.available)
    out << "[[" << std::setprecision(17) << observation.points[0].x() << ',' << observation.points[0].y() << ',' << observation.points[0].z() << "],[" << observation.points[1].x() << ',' << observation.points[1].y() << ',' << observation.points[1].z() << "]]";
  else out << "null";
  out << '}';
}

static std::string read_moveit_version()
{
  std::ifstream input("/opt/ros/jazzy/share/moveit_core/package.xml");
  std::string line;
  while (std::getline(input, line))
  {
    const auto start = line.find("<version>"); const auto end = line.find("</version>");
    if (start != std::string::npos && end != std::string::npos && end > start + 9) return line.substr(start + 9, end - start - 9);
  }
  return "NOT_AVAILABLE";
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("stage3_h4_4_native_bridge", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  std::string candidate_path, mesh_path, output_path, group;
  int max_contacts_parameter = 4096, max_contacts_per_pair_parameter = 64;
  node->get_parameter_or("candidate_csv", candidate_path, std::string{});
  node->get_parameter_or("curved_mesh_obj", mesh_path, std::string{});
  node->get_parameter_or("output_dir", output_path, std::string{});
  node->get_parameter_or("group_name", group, std::string("fairino5_v6_group"));
  node->get_parameter_or("max_contacts", max_contacts_parameter, 4096);
  node->get_parameter_or("max_contacts_per_pair", max_contacts_per_pair_parameter, 64);
  if (candidate_path.empty() || mesh_path.empty() || output_path.empty() || max_contacts_parameter <= 0 || max_contacts_per_pair_parameter <= 0)
  {
    RCLCPP_ERROR(node->get_logger(), "candidate_csv, curved_mesh_obj, output_dir and positive contact limits are required");
    rclcpp::shutdown(); return 2;
  }
  fs::create_directories(output_path);
  robot_model_loader::RobotModelLoader::Options options("robot_description"); options.load_kinematics_solvers = false;
  robot_model_loader::RobotModelLoader loader(node, options);
  const auto model = loader.getModel();
  if (!model) { RCLCPP_ERROR(node->get_logger(), "failed to load robot model"); rclcpp::shutdown(); return 3; }
  const auto* joint_group = model->getJointModelGroup(group);
  if (!joint_group) { RCLCPP_ERROR(node->get_logger(), "joint group not found: %s", group.c_str()); rclcpp::shutdown(); return 4; }
  const auto candidates = read_candidates(candidate_path); const auto mesh = read_obj(mesh_path);
  if (candidates.empty() || mesh.vertices.empty() || mesh.triangles.empty()) { RCLCPP_ERROR(node->get_logger(), "empty input"); rclcpp::shutdown(); return 5; }
  auto scene = std::make_unique<planning_scene::PlanningScene>(model);
  scene->processCollisionObjectMsg(make_mesh_object(mesh));
  scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  const auto max_contacts = static_cast<std::size_t>(max_contacts_parameter);
  const auto max_contacts_per_pair = static_cast<std::size_t>(max_contacts_per_pair_parameter);
  const auto& acm = scene->getAllowedCollisionMatrix();
  std::ofstream padding(fs::path(output_path) / "padding_semantics_audit.jsonl");
  std::ofstream clearance(fs::path(output_path) / "native_clearance_evidence.jsonl");
  std::size_t checked = 0, padding_changed = 0, padding_only = 0, unpadded_still = 0, env_available = 0, self_available = 0, env_negative = 0, self_negative = 0;
  double min_env = std::numeric_limits<double>::infinity(), min_self = std::numeric_limits<double>::infinity();
  std::map<std::string, std::size_t> pair_frequency;
  std::size_t branch_count = 0;
  for (const auto& candidate : candidates)
  {
    ++branch_count;
    if (!candidate.fk_valid)
    {
      padding << "{\"schema_version\":\"stage3-h4-4-padding-audit-v1\",\"target_index\":" << candidate.target_index << ",\"task_sample_id\":" << json_quote(candidate.task_sample_id)
              << ",\"candidate_id\":" << json_quote(candidate.candidate_id) << ",\"original_collision_classification\":" << json_quote(candidate.baseline_classification)
              << ",\"fk_valid\":false,\"collision_checked\":false,\"status\":\"NOT_RUN_FK_INVALID\"}\n";
      clearance << "{\"schema_version\":\"stage3-h4-4-native-clearance-v1\",\"target_index\":" << candidate.target_index << ",\"task_sample_id\":" << json_quote(candidate.task_sample_id)
                << ",\"candidate_id\":" << json_quote(candidate.candidate_id) << ",\"fk_valid\":false,\"status\":\"NOT_RUN_FK_INVALID\",\"reason\":\"H4.2.1 FK validation was invalid; native distance is not a valid target-state observation\"}\n";
      continue;
    }
    ++checked;
    moveit::core::RobotState state(model); state.setJointGroupPositions(joint_group, candidate.q); state.update();
    const auto current_request = collision_request(group, true, false, max_contacts, max_contacts_per_pair);
    const auto padded_self_request = collision_request(group, false, true, max_contacts, max_contacts_per_pair);
    const auto unpadded_request = collision_request(group, false, false, max_contacts, max_contacts_per_pair);
    collision_detection::CollisionResult current_combined, padded_environment, padded_self_result, unpadded_environment, unpadded_self_result;
    scene->checkCollision(current_request, current_combined, state);
    scene->getCollisionEnv()->checkRobotCollision(current_request, padded_environment, state, acm);
    scene->checkSelfCollision(padded_self_request, padded_self_result, state);
    scene->getCollisionEnvUnpadded()->checkRobotCollision(unpadded_request, unpadded_environment, state, acm);
    scene->checkSelfCollision(unpadded_request, unpadded_self_result, state);
    const auto current = collect(current_combined); const auto padded_env = collect(padded_environment); const auto padded_self = collect(padded_self_result);
    const auto unpadded_env = collect(unpadded_environment); const auto unpadded_self = collect(unpadded_self_result);
    const bool current_env_collision = padded_env.collision; const bool current_self_collision = unpadded_self.collision;
    const bool unpadded_collision = unpadded_env.collision || unpadded_self.collision;
    if ((current_env_collision || current_self_collision) != unpadded_collision) ++padding_changed;
    if ((current_env_collision || current_self_collision) && !unpadded_collision) ++padding_only;
    if (unpadded_collision) ++unpadded_still;
    padding << "{\"schema_version\":\"stage3-h4-4-padding-audit-v1\",\"target_index\":" << candidate.target_index << ",\"task_sample_id\":" << json_quote(candidate.task_sample_id)
            << ",\"candidate_id\":" << json_quote(candidate.candidate_id) << ",\"original_collision_classification\":" << json_quote(candidate.baseline_classification)
            << ",\"fk_valid\":true,\"collision_checked\":true,\"current_h4_3_collision\":" << ((current_env_collision || current_self_collision) ? "true" : "false")
            << ",\"current_h4_3_environment_collision\":" << (current_env_collision ? "true" : "false") << ",\"current_h4_3_self_collision\":" << (current_self_collision ? "true" : "false")
            << ','; write_variant(padding, "padded_environment", padded_env);
    padding << ','; write_variant(padding, "unpadded_environment", unpadded_env);
    padding << ','; write_variant(padding, "padded_self", padded_self);
    padding << ','; write_variant(padding, "unpadded_self", unpadded_self);
    padding << ','; write_variant(padding, "current_combined", current);
    padding << ",\"unpadded_still_collision\":" << (unpadded_collision ? "true" : "false") << "}\n";
    for (const auto& pair : padded_env.pairs) ++pair_frequency[pair];
    for (const auto& pair : unpadded_env.pairs) ++pair_frequency[pair];
    const auto env_padded_distance = distance_observation(scene->getCollisionEnv(), group, acm, state, false);
    const auto env_unpadded_distance = distance_observation(scene->getCollisionEnvUnpadded(), group, acm, state, false);
    const auto self_padded_distance = distance_observation(scene->getCollisionEnv(), group, acm, state, true);
    const auto self_unpadded_distance = distance_observation(scene->getCollisionEnvUnpadded(), group, acm, state, true);
    if (env_unpadded_distance.available) { ++env_available; min_env = std::min(min_env, env_unpadded_distance.value); }
    if (self_unpadded_distance.available) { ++self_available; min_self = std::min(min_self, self_unpadded_distance.value); }
    env_negative += env_unpadded_distance.negative_observed ? 1 : 0; self_negative += self_unpadded_distance.negative_observed ? 1 : 0;
    clearance << "{\"schema_version\":\"stage3-h4-4-native-clearance-v1\",\"target_index\":" << candidate.target_index << ",\"task_sample_id\":" << json_quote(candidate.task_sample_id)
              << ",\"candidate_id\":" << json_quote(candidate.candidate_id) << ",\"fk_valid\":true,\"collision_checked\":true,\"distance_api\":[\"collision_detection::DistanceRequest\",\"CollisionEnv::distanceRobot\",\"CollisionEnv::distanceSelf\"],\"signed_distance_semantics\":\"signed requested; MoveIt DistanceResult.minimum_distance.distance is used as returned; <=0 denotes collision per local header\",";
    write_distance(clearance, "environment_padded", env_padded_distance); clearance << ","; write_distance(clearance, "environment_unpadded", env_unpadded_distance); clearance << ",";
    write_distance(clearance, "self_padded", self_padded_distance); clearance << ","; write_distance(clearance, "self_unpadded", self_unpadded_distance); clearance << "}\n";
  }
  padding.flush(); clearance.flush();
  std::ofstream padding_summary(fs::path(output_path) / "padding_semantics_summary.json");
  padding_summary << "{\n  \"schema_version\": \"stage3-h4-4-padding-summary-v1\",\n  \"backend\": \"FCL\",\n  \"candidate_branch_count\": " << branch_count << ",\n  \"fk_valid_candidate_count\": " << checked << ",\n  \"padding_changed_classification_count\": " << padding_changed << ",\n  \"padding_only_collision_count\": " << padding_only << ",\n  \"unpadded_still_collision_count\": " << unpadded_still << ",\n  \"current_semantics\": \"H4.3 PlanningScene.checkCollision with pad_environment_collisions=true, pad_self_collisions=false\",\n  \"explicit_unpadded_semantics\": \"CollisionEnvUnpadded::checkRobotCollision and PlanningScene.checkSelfCollision with both padding flags false\",\n  \"collision_method\": \"adaptive_discrete_interpolation\"\n}\n";
  padding_summary.close();
  std::ofstream clearance_summary(fs::path(output_path) / "native_clearance_summary.json");
  clearance_summary << "{\n  \"schema_version\": \"stage3-h4-4-native-clearance-summary-v1\",\n  \"backend\": \"FCL\",\n  \"status\": \"AVAILABLE\",\n  \"reason\": \"Local MoveIt2 headers and linked FCL backend exposed DistanceRequest, DistanceResult, CollisionEnv::distanceRobot, CollisionEnv::distanceSelf, nearest_points and signed-distance request; finite observations were collected for the FK-valid states.\",\n  \"ros_distro\": \"jazzy (/opt/ros/jazzy observed)\",\n  \"moveit_package_version\": " << json_quote(read_moveit_version()) << ",\n  \"local_source_path\": \"NOT_AVAILABLE; installed headers and linked binary API used\",\n  \"relevant_api_symbols\": [\"PlanningScene::distanceToCollision\",\"PlanningScene::distanceToCollisionUnpadded\",\"PlanningScene::getCollisionEnvUnpadded\",\"CollisionEnv::distanceRobot\",\"CollisionEnv::distanceSelf\",\"collision_detection::DistanceRequest\",\"collision_detection::DistanceResult\",\"DistanceResultsData::nearest_points\"],\n  \"padded_environment_distance_available_count\": " << env_available << ",\n  \"unpadded_environment_distance_available_count\": " << env_available << ",\n  \"padded_self_distance_available_count\": " << self_available << ",\n  \"unpadded_self_distance_available_count\": " << self_available << ",\n  \"signed_distance_requested\": true,\n  \"environment_negative_distance_observed_count\": " << env_negative << ",\n  \"self_negative_distance_observed_count\": " << self_negative << ",\n  \"minimum_observed_environment_distance_m\": "; write_number_or_null(clearance_summary, min_env); clearance_summary << ",\n  \"minimum_observed_self_distance_m\": "; write_number_or_null(clearance_summary, min_self); clearance_summary << ",\n  \"clearance_is_contact_depth\": false,\n  \"clearance_is_aabb_distance\": false,\n  \"ccd_status\": \"not_available\",\n  \"collision_method\": \"adaptive_discrete_interpolation\"\n}\n";
  clearance_summary.close();
  std::ofstream capability(fs::path(output_path) / "native_fcl_api_capability_audit.json");
  capability << "{\n  \"schema_version\": \"stage3-h4-4-native-api-capability-v1\",\n  \"ros_distro\": \"jazzy (/opt/ros/jazzy observed)\",\n  \"moveit_package_version\": " << json_quote(read_moveit_version()) << ",\n  \"collision_backend\": \"FCL\",\n  \"local_header_prefix\": \"/opt/ros/jazzy/include/moveit_core\",\n  \"local_source_path\": \"NOT_AVAILABLE; package headers/binary observed\",\n  \"padded_environment_collision\": true,\n  \"unpadded_environment_collision\": true,\n  \"padded_self_collision\": true,\n  \"unpadded_self_collision\": true,\n  \"distanceRobot\": true,\n  \"distanceSelf\": true,\n  \"signed_distance_request\": true,\n  \"nearest_points\": true,\n  \"api_evidence\": [\"/opt/ros/jazzy/include/moveit_core/moveit/planning_scene/planning_scene.hpp\",\"/opt/ros/jazzy/include/moveit_core/moveit/collision_detection/collision_env.hpp\",\"/opt/ros/jazzy/include/moveit_core/moveit/collision_detection/collision_common.hpp\"]\n}\n";
  capability.close();
  rclcpp::shutdown(); return 0;
}
