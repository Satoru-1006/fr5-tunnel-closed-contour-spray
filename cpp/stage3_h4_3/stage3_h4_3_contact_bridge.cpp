#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
  double fk_translation_error_m = 0.0;
  double fk_orientation_error_rad = 0.0;
  std::string joint_limit_state;
  std::vector<double> q;
};

struct ObjMesh
{
  std::vector<Eigen::Vector3d> vertices;
  std::vector<Eigen::Vector3i> triangles;
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
      if (quoted && i + 1 < line.size() && line[i + 1] == '"')
      {
        field.push_back('"');
        ++i;
      }
      else
      {
        quoted = !quoted;
      }
    }
    else if (c == ',' && !quoted)
    {
      fields.push_back(field);
      field.clear();
    }
    else
    {
      field.push_back(c);
    }
  }
  fields.push_back(field);
  if (!fields.empty() && !fields.back().empty() && fields.back().back() == '\r')
    fields.back().pop_back();
  return fields;
}

static std::string csv_escape(const std::string& value)
{
  if (value.find_first_of(",\"\n") == std::string::npos)
    return value;
  std::string result = "\"";
  for (const char c : value)
    result += (c == '"') ? "\"\"" : std::string(1, c);
  result += '"';
  return result;
}

static double number(const std::string& value)
{
  try
  {
    return std::stod(value);
  }
  catch (...)
  {
    return 0.0;
  }
}

static std::vector<Candidate> read_candidates(const fs::path& path)
{
  std::ifstream input(path);
  std::string line;
  if (!std::getline(input, line))
    return {};
  const auto header = split_csv(line);
  std::map<std::string, std::size_t> column;
  for (std::size_t i = 0; i < header.size(); ++i)
    column[header[i]] = i;
  std::vector<Candidate> result;
  while (std::getline(input, line))
  {
    if (line.empty())
      continue;
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
    candidate.fk_translation_error_m = number(get("fk_translation_error_m"));
    candidate.fk_orientation_error_rad = number(get("fk_orientation_error_rad"));
    candidate.joint_limit_state = get("joint_limit_state");
    for (int i = 0; i < 6; ++i)
      candidate.q.push_back(number(get("q" + std::to_string(i))));
    if (candidate.q.size() == 6 && !candidate.task_sample_id.empty())
      result.push_back(std::move(candidate));
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
      double x{}, y{}, z{};
      stream >> x >> y >> z;
      mesh.vertices.emplace_back(x, y, z);
    }
    else if (tag == "f")
    {
      std::vector<int> indices;
      std::string token;
      while (stream >> token)
      {
        const auto slash = token.find('/');
        const auto first = token.substr(0, slash);
        const int index = std::stoi(first);
        indices.push_back(index > 0 ? index - 1 : static_cast<int>(mesh.vertices.size()) + index);
      }
      for (std::size_t i = 1; i + 1 < indices.size(); ++i)
        mesh.triangles.emplace_back(indices[0], indices[i], indices[i + 1]);
    }
  }
  return mesh;
}

static moveit_msgs::msg::CollisionObject make_mesh_object(const ObjMesh& mesh, const std::string& object_id)
{
  moveit_msgs::msg::CollisionObject object;
  object.header.frame_id = "base_link";
  object.id = object_id;
  shape_msgs::msg::Mesh shape;
  for (const auto& vertex : mesh.vertices)
  {
    geometry_msgs::msg::Point point;
    point.x = vertex.x();
    point.y = vertex.y();
    point.z = vertex.z();
    shape.vertices.push_back(point);
  }
  for (const auto& triangle : mesh.triangles)
  {
    shape_msgs::msg::MeshTriangle face;
    face.vertex_indices = {static_cast<unsigned int>(triangle.x()), static_cast<unsigned int>(triangle.y()), static_cast<unsigned int>(triangle.z())};
    shape.triangles.push_back(face);
  }
  object.meshes.push_back(shape);
  geometry_msgs::msg::Pose pose;
  pose.orientation.w = 1.0;
  object.mesh_poses.push_back(pose);
  object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return object;
}

static std::string json_quote(const std::string& value)
{
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value)
  {
    if (c == '"') out << "\\\"";
    else if (c == '\\') out << "\\\\";
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else out << c;
  }
  out << '"';
  return out.str();
}

static bool is_world(const collision_detection::Contact& contact, bool first)
{
  return (first ? contact.body_type_1 : contact.body_type_2) == collision_detection::BodyTypes::WORLD_OBJECT;
}

static void write_contact_json(std::ofstream& out, const Candidate& candidate, const collision_detection::Contact& contact,
                               const std::string& category, std::size_t contact_count, std::size_t contact_index)
{
  const std::string body_1 = contact.body_name_1;
  const std::string body_2 = contact.body_name_2;
  const bool env = category == "environment_collision";
  const std::string robot_link = env ? (is_world(contact, true) ? body_2 : body_1) : "";
  const std::string environment = env ? (is_world(contact, true) ? body_1 : body_2) : "";
  out << "{\"schema_version\":\"stage3-h4-3-contact-v1\",\"target_index\":" << candidate.target_index
      << ",\"task_sample_id\":" << json_quote(candidate.task_sample_id)
      << ",\"candidate_id\":" << json_quote(candidate.candidate_id)
      << ",\"target_classification\":" << json_quote(candidate.target_classification)
      << ",\"baseline_collision_category\":" << json_quote(candidate.baseline_classification)
      << ",\"collision_category\":" << json_quote(category)
      << ",\"robot_link\":" << json_quote(robot_link)
      << ",\"environment_object\":" << json_quote(environment)
      << ",\"body_1\":" << json_quote(body_1)
      << ",\"body_2\":" << json_quote(body_2)
      << ",\"contact_count\":" << contact_count
      << ",\"contact_index\":" << contact_index
      << ",\"contact_point_xyz_m\":[" << std::setprecision(17) << contact.pos.x() << ',' << contact.pos.y() << ',' << contact.pos.z() << ']'
      << ",\"normal_xyz\":[" << contact.normal.x() << ',' << contact.normal.y() << ',' << contact.normal.z() << ']'
      << ",\"penetration_depth_m\":" << contact.depth
      << ",\"percent_interpolation\":" << contact.percent_interpolation
      << ",\"backend\":\"FCL\",\"collision_api\":\"PlanningScene.checkCollision + checkSelfCollision\"}\n";
}

static void write_pair_json(std::ofstream& out, const Candidate& candidate, const collision_detection::Contact& contact,
                            const std::string& category, std::size_t contact_count, std::size_t contact_index)
{
  const bool env = category == "environment_collision";
  const std::string first = contact.body_name_1;
  const std::string second = contact.body_name_2;
  const std::string pair_1 = env && is_world(contact, true) ? second : first;
  const std::string pair_2 = env && is_world(contact, true) ? first : second;
  out << "{\"schema_version\":\"stage3-h4-3-pair-v1\",\"target_index\":" << candidate.target_index
      << ",\"task_sample_id\":" << json_quote(candidate.task_sample_id)
      << ",\"candidate_id\":" << json_quote(candidate.candidate_id)
      << ",\"target_classification\":" << json_quote(candidate.target_classification)
      << ",\"collision_category\":" << json_quote(category)
      << ",\"link_or_object_1\":" << json_quote(pair_1)
      << ",\"link_or_object_2\":" << json_quote(pair_2)
      << ",\"contact_count\":" << contact_count
      << ",\"contact_index\":" << contact_index
      << ",\"backend\":\"FCL\",\"contact_point_xyz_m\":[" << std::setprecision(17) << contact.pos.x() << ',' << contact.pos.y() << ',' << contact.pos.z() << ']'
      << ",\"penetration_depth_m\":" << contact.depth << "}\n";
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("stage3_h4_3_contact_bridge", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  std::string candidate_path, mesh_path, output_path, group;
  int max_contacts_parameter = 4096;
  int max_contacts_per_pair_parameter = 64;
  node->get_parameter_or("candidate_csv", candidate_path, std::string{});
  node->get_parameter_or("curved_mesh_obj", mesh_path, std::string{});
  node->get_parameter_or("output_dir", output_path, std::string{});
  node->get_parameter_or("group_name", group, std::string("fairino5_v6_group"));
  node->get_parameter_or("max_contacts", max_contacts_parameter, 4096);
  node->get_parameter_or("max_contacts_per_pair", max_contacts_per_pair_parameter, 64);
  if (candidate_path.empty() || mesh_path.empty() || output_path.empty() || max_contacts_parameter <= 0 || max_contacts_per_pair_parameter <= 0)
  {
    RCLCPP_ERROR(node->get_logger(), "candidate_csv, curved_mesh_obj, output_dir and positive contact limits are required");
    rclcpp::shutdown();
    return 2;
  }
  fs::create_directories(output_path);
  robot_model_loader::RobotModelLoader::Options options("robot_description");
  options.load_kinematics_solvers = false;
  robot_model_loader::RobotModelLoader loader(node, options);
  const auto model = loader.getModel();
  if (!model)
  {
    RCLCPP_ERROR(node->get_logger(), "failed to load robot model");
    rclcpp::shutdown();
    return 3;
  }
  const auto* joint_group = model->getJointModelGroup(group);
  if (!joint_group)
  {
    RCLCPP_ERROR(node->get_logger(), "joint group not found: %s", group.c_str());
    rclcpp::shutdown();
    return 4;
  }
  const auto candidates = read_candidates(candidate_path);
  const auto mesh = read_obj(mesh_path);
  if (candidates.empty() || mesh.vertices.empty() || mesh.triangles.empty())
  {
    RCLCPP_ERROR(node->get_logger(), "empty input candidates=%zu vertices=%zu triangles=%zu", candidates.size(), mesh.vertices.size(), mesh.triangles.size());
    rclcpp::shutdown();
    return 5;
  }
  auto scene = std::make_unique<planning_scene::PlanningScene>(model);
  scene->processCollisionObjectMsg(make_mesh_object(mesh, "fixture_surface:fixture_curved_cylinder_patch"));
  scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());

  const auto max_contacts = static_cast<std::size_t>(max_contacts_parameter);
  const auto max_contacts_per_pair = static_cast<std::size_t>(max_contacts_per_pair_parameter);
  const fs::path root(output_path);
  std::ofstream contacts(root / "curved_contact_evidence.jsonl");
  std::ofstream self_pairs(root / "self_collision_pairs.jsonl");
  std::ofstream env_pairs(root / "environment_collision_pairs.jsonl");
  std::ofstream summary(root / "native_contact_summary.json");
  std::map<std::string, std::size_t> robot_links;
  std::map<std::string, std::size_t> environment_objects;
  std::map<std::string, std::size_t> self_pair_counts;
  std::map<std::string, std::size_t> environment_pair_counts;
  std::map<std::string, std::size_t> baseline_counts;
  std::map<std::string, std::size_t> native_counts;
  std::size_t total_contacts = 0;
  std::size_t checked = 0;
  std::size_t joint_limit_failures = 0;
  std::size_t category_mismatches = 0;
  for (const auto& candidate : candidates)
  {
    moveit::core::RobotState state(model);
    state.setJointGroupPositions(joint_group, candidate.q);
    state.update();
    const bool bounds = state.satisfiesBounds(joint_group);
    if (!bounds) ++joint_limit_failures;
    collision_detection::CollisionRequest request;
    request.group_name = group;
    request.contacts = true;
    request.max_contacts = max_contacts;
    request.max_contacts_per_pair = max_contacts_per_pair;
    request.pad_environment_collisions = true;
    request.pad_self_collisions = false;
    request.verbose = true;
    collision_detection::CollisionResult full;
    full.clear();
    scene->checkCollision(request, full, state);
    collision_detection::CollisionResult self;
    self.clear();
    scene->checkSelfCollision(request, self, state);
    ++checked;
    bool self_collision = self.collision;
    bool environment_collision = false;
    std::size_t candidate_contacts = 0;
    std::size_t contact_index = 0;
    for (const auto& entry : full.contacts)
    {
      for (const auto& contact : entry.second)
      {
        ++candidate_contacts;
        const bool first_world = is_world(contact, true);
        const bool second_world = is_world(contact, false);
        const bool is_env = first_world || second_world;
        const std::string category = is_env ? "environment_collision" : "self_collision";
        environment_collision = environment_collision || is_env;
        self_collision = self_collision || !is_env;
        write_contact_json(contacts, candidate, contact, category, full.contact_count, contact_index);
        if (is_env)
        {
          const std::string robot = first_world ? contact.body_name_2 : contact.body_name_1;
          const std::string object = first_world ? contact.body_name_1 : contact.body_name_2;
          ++robot_links[robot];
          ++environment_objects[object];
          ++environment_pair_counts[robot + "|" + object];
          write_pair_json(env_pairs, candidate, contact, category, full.contact_count, contact_index);
        }
        else
        {
          const std::string a = std::min(contact.body_name_1, contact.body_name_2);
          const std::string b = std::max(contact.body_name_1, contact.body_name_2);
          ++self_pair_counts[a + "|" + b];
          write_pair_json(self_pairs, candidate, contact, category, full.contact_count, contact_index);
        }
        ++contact_index;
      }
    }
    total_contacts += candidate_contacts;
    const std::string native_category = environment_collision && self_collision
      ? "IK_FOUND_BOTH_COLLISION"
      : (environment_collision ? "IK_FOUND_ENV_COLLISION" : (self_collision ? "IK_FOUND_SELF_COLLISION" : "REACHABLE_COLLISION_FREE"));
    ++baseline_counts[candidate.baseline_classification];
    ++native_counts[native_category];
    if (native_category != candidate.baseline_classification) ++category_mismatches;
  }
  contacts.flush();
  self_pairs.flush();
  env_pairs.flush();
  summary << "{\n"
          << "  \"schema_version\": \"stage3-h4-3-native-contact-summary-v1\",\n"
          << "  \"backend\": \"FCL\",\n"
          << "  \"collision_api\": [\"collision_detection::CollisionRequest\", \"PlanningScene::checkCollision\", \"PlanningScene::checkSelfCollision\", \"CollisionResult.contacts\"],\n"
          << "  \"contacts_requested\": true,\n"
          << "  \"max_contacts\": " << max_contacts << ",\n"
          << "  \"max_contacts_per_pair\": " << max_contacts_per_pair << ",\n"
          << "  \"candidate_count\": " << candidates.size() << ",\n"
          << "  \"collision_checked_count\": " << checked << ",\n"
          << "  \"total_contact_count\": " << total_contacts << ",\n"
          << "  \"joint_limit_failure_count\": " << joint_limit_failures << ",\n"
          << "  \"category_mismatch_count\": " << category_mismatches << ",\n"
          << "  \"mesh_vertex_count\": " << mesh.vertices.size() << ",\n"
          << "  \"mesh_triangle_count\": " << mesh.triangles.size() << ",\n"
          << "  \"ccd_status\": \"not_available\",\n"
          << "  \"clearance_status\": \"not_available\",\n"
          << "  \"clearance_m\": null,\n"
          << "  \"collision_method\": \"adaptive_discrete_interpolation\",\n"
          << "  \"baseline_category_counts\": {";
  bool first = true;
  for (const auto& item : baseline_counts) { if (!first) summary << ','; first = false; summary << json_quote(item.first) << ':' << item.second; }
  summary << "},\n  \"native_category_counts\": {";
  first = true;
  for (const auto& item : native_counts) { if (!first) summary << ','; first = false; summary << json_quote(item.first) << ':' << item.second; }
  summary << "},\n  \"robot_link_contact_frequency\": {";
  first = true;
  for (const auto& item : robot_links) { if (!first) summary << ','; first = false; summary << json_quote(item.first) << ':' << item.second; }
  summary << "},\n  \"environment_object_contact_frequency\": {";
  first = true;
  for (const auto& item : environment_objects) { if (!first) summary << ','; first = false; summary << json_quote(item.first) << ':' << item.second; }
  summary << "},\n  \"self_collision_pair_frequency\": {";
  first = true;
  for (const auto& item : self_pair_counts) { if (!first) summary << ','; first = false; summary << json_quote(item.first) << ':' << item.second; }
  summary << "},\n  \"environment_pair_frequency\": {";
  first = true;
  for (const auto& item : environment_pair_counts) { if (!first) summary << ','; first = false; summary << json_quote(item.first) << ':' << item.second; }
  summary << "}\n}\n";
  summary.flush();
  contacts.close();
  self_pairs.close();
  env_pairs.close();
  summary.close();
  rclcpp::shutdown();
  return 0;
}
