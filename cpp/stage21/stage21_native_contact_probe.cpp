#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection_bullet/collision_detector_allocator_bullet.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

namespace fs = std::filesystem;

struct Candidate
{
  int waypoint = -1;
  std::string id;
  std::vector<double> q;
};

struct PoseRow
{
  double x{}, y{}, z{}, nx{}, ny{}, nz{};
};

struct Variant
{
  std::string name;
  bool include_floor = true;
  bool padded = true;
  bool acm_disabled = false;
  bool bullet = false;
  bool include_world = true;
  std::string allowed_body_1;
  std::string allowed_body_2;
  std::string scope = "full";
};

struct CollisionSummary
{
  bool full = false;
  bool self = false;
  bool robot_world = false;
  bool both = false;
  std::size_t contact_count = 0;
  std::vector<collision_detection::Contact> contacts;
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

static std::vector<double> json_vector(std::string value)
{
  std::vector<double> result;
  const auto begin = value.find('[');
  const auto end = value.rfind(']');
  if (begin != std::string::npos)
    value = value.substr(begin + 1, end > begin ? end - begin - 1 : std::string::npos);
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ','))
    result.push_back(number(token));
  return result;
}

static std::vector<PoseRow> read_poses(const fs::path& path)
{
  std::ifstream input(path);
  std::string line;
  std::getline(input, line);
  const auto header = split_csv(line);
  std::map<std::string, std::size_t> column;
  for (std::size_t i = 0; i < header.size(); ++i)
    column[header[i]] = i;
  std::vector<PoseRow> result;
  while (std::getline(input, line))
  {
    if (line.empty())
      continue;
    const auto fields = split_csv(line);
    auto get = [&](const std::string& name) { return number(fields.at(column.at(name))); };
    result.push_back({get("x"), get("y"), get("z"), get("nx"), get("ny"), get("nz")});
  }
  return result;
}

static std::vector<Candidate> read_candidates(const fs::path& path)
{
  std::ifstream input(path);
  std::string line;
  std::getline(input, line);
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
    const auto q = json_vector(fields.at(column.at("joint_values")));
    if (q.size() != 6)
      continue;
    result.push_back({static_cast<int>(number(fields.at(column.at("waypoint_index")))), fields.at(column.at("candidate_id")), q});
  }
  return result;
}

static Eigen::Vector3d v3(const PoseRow& p, bool normal = false)
{
  return normal ? Eigen::Vector3d(p.nx, p.ny, p.nz) : Eigen::Vector3d(p.x, p.y, p.z);
}

static moveit_msgs::msg::CollisionObject make_box(const std::string& id, const Eigen::Vector3d& dims,
                                                   const Eigen::Vector3d& center, const Eigen::Matrix3d& rotation)
{
  moveit_msgs::msg::CollisionObject object;
  object.header.frame_id = "base_link";
  object.id = id;
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  primitive.dimensions = {dims.x(), dims.y(), dims.z()};
  geometry_msgs::msg::Pose pose;
  pose.position.x = center.x();
  pose.position.y = center.y();
  pose.position.z = center.z();
  const Eigen::Quaterniond quaternion(rotation);
  pose.orientation.x = quaternion.x();
  pose.orientation.y = quaternion.y();
  pose.orientation.z = quaternion.z();
  pose.orientation.w = quaternion.w();
  object.primitives.push_back(primitive);
  object.primitive_poses.push_back(pose);
  object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return object;
}

static std::unique_ptr<planning_scene::PlanningScene> build_scene(
    const moveit::core::RobotModelConstPtr& model, const std::vector<PoseRow>& poses, bool include_floor, bool bullet,
    bool include_world)
{
  auto scene = std::make_unique<planning_scene::PlanningScene>(model);
  const double stand_off = 0.260;
  const double wall_thickness = 0.040;
  const double y_thickness = 1.10;
  std::vector<Eigen::Vector3d> wall_points;
  std::vector<Eigen::Vector3d> normals;
  wall_points.reserve(poses.size());
  normals.reserve(poses.size());
  for (const auto& row : poses)
  {
    Eigen::Vector3d normal = v3(row, true);
    normal.normalize();
    wall_points.push_back(v3(row) + stand_off * normal);
    normals.push_back(normal);
  }
  if (include_world)
  {
    std::size_t loop_size = wall_points.size();
    for (std::size_t i = 16; i < wall_points.size(); ++i)
    {
      if ((wall_points[i] - wall_points.front()).norm() < 1e-7)
      {
        loop_size = i;
        break;
      }
    }
    for (std::size_t start = 0; start + 1 < loop_size; ++start)
    {
      const std::size_t end = std::min(start + 1, loop_size - 1);
      const Eigen::Vector3d segment = wall_points[end] - wall_points[start];
      const double length = segment.norm();
      if (length <= 1e-9)
        continue;
      const Eigen::Vector3d x_axis = segment / length;
      const Eigen::Vector3d y_axis(0.0, 1.0, 0.0);
      Eigen::Vector3d z_axis = x_axis.cross(y_axis).normalized();
      if (z_axis.dot(normals[start]) < 0.0)
        z_axis = -z_axis;
      Eigen::Matrix3d rotation;
      rotation.col(0) = x_axis;
      rotation.col(1) = y_axis;
      rotation.col(2) = z_axis;
      const Eigen::Vector3d center = (wall_points[start] + wall_points[end]) * 0.5 + normals[start] * (wall_thickness * 0.5);
      scene->processCollisionObjectMsg(make_box("horseshoe_wall_" + std::to_string(start), Eigen::Vector3d(length + wall_thickness, y_thickness, wall_thickness), center, rotation));
    }
    if (include_floor)
    {
      double min_x = wall_points.front().x(), max_x = min_x;
      double mean_y = 0.0;
      for (const auto& point : wall_points)
      {
        min_x = std::min(min_x, point.x());
        max_x = std::max(max_x, point.x());
        mean_y += point.y();
      }
      mean_y /= static_cast<double>(wall_points.size());
      scene->processCollisionObjectMsg(make_box("tunnel_floor", Eigen::Vector3d(max_x - min_x + wall_thickness, y_thickness, wall_thickness), Eigen::Vector3d((max_x + min_x) * 0.5, mean_y, -0.220), Eigen::Matrix3d::Identity()));
    }
  }
  if (bullet)
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create());
  else
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  return scene;
}

static CollisionSummary check_state(planning_scene::PlanningScene& scene, moveit::core::RobotState& state,
                                    const std::string& group, bool padded, bool distance,
                                    std::size_t max_contacts, std::size_t max_contacts_per_pair)
{
  collision_detection::CollisionRequest request;
  request.group_name = group;
  request.contacts = true;
  request.max_contacts = max_contacts;
  request.max_contacts_per_pair = max_contacts_per_pair;
  request.is_done = nullptr;
  request.distance = distance;
  request.detailed_distance = distance;
  request.verbose = true;
  request.pad_environment_collisions = padded;
  request.pad_self_collisions = false;
  collision_detection::CollisionResult full;
  full.clear();
  scene.checkCollision(request, full, state);
  collision_detection::CollisionResult self;
  self.clear();
  scene.checkSelfCollision(request, self, state);
  CollisionSummary summary;
  summary.full = full.collision;
  summary.self = self.collision;
  summary.contact_count = full.contact_count;
  for (const auto& pair : full.contacts)
  {
    for (const auto& contact : pair.second)
    {
      summary.contacts.push_back(contact);
      const bool first_robot = contact.body_type_1 != collision_detection::BodyTypes::WORLD_OBJECT;
      const bool second_robot = contact.body_type_2 != collision_detection::BodyTypes::WORLD_OBJECT;
      const bool first_world = contact.body_type_1 == collision_detection::BodyTypes::WORLD_OBJECT;
      const bool second_world = contact.body_type_2 == collision_detection::BodyTypes::WORLD_OBJECT;
      summary.robot_world = summary.robot_world || ((first_robot && second_world) || (second_robot && first_world));
      summary.self = summary.self || (first_robot && second_robot);
    }
  }
  summary.robot_world = summary.robot_world || (summary.full && !summary.self);
  summary.both = summary.self && summary.robot_world;
  return summary;
}

static void write_contact(std::ofstream& out, const std::string& variant, const Candidate& candidate,
                          const collision_detection::Contact& contact, std::size_t raw_contact_count,
                          std::size_t contact_index, const std::string& export_status)
{
  const bool first_robot = contact.body_type_1 != collision_detection::BodyTypes::WORLD_OBJECT;
  const bool second_robot = contact.body_type_2 != collision_detection::BodyTypes::WORLD_OBJECT;
  const bool first_world = contact.body_type_1 == collision_detection::BodyTypes::WORLD_OBJECT;
  const bool second_world = contact.body_type_2 == collision_detection::BodyTypes::WORLD_OBJECT;
  const std::string type = (first_robot && second_robot) ? "self" : ((first_robot && second_world) || (second_robot && first_world) ? "robot_world" : "unknown");
  out << variant << ',' << candidate.waypoint << ',' << csv_escape(candidate.id) << ','
      << csv_escape(contact.body_name_1) << ',' << csv_escape(contact.body_name_2) << ',' << type << ','
      << contact.pos.x() << ',' << contact.pos.y() << ',' << contact.pos.z() << ','
      << contact.normal.x() << ',' << contact.normal.y() << ',' << contact.normal.z() << ','
      << std::setprecision(17) << contact.depth << ',' << contact.percent_interpolation << ','
      << raw_contact_count << ',' << contact_index << ',' << export_status << '\n';
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("stage21_native_contact_probe", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  std::string candidate_path;
  std::string pose_path;
  std::string output_path;
  std::string group;
  bool export_all_contacts = false;
  bool distance_request = true;
  bool bullet_only = false;
  std::string variant_filter;
  int max_contacts_parameter = 4096;
  int max_contacts_per_pair_parameter = 64;
  node->get_parameter_or("candidate_csv", candidate_path, std::string{});
  node->get_parameter_or("pose_csv", pose_path, std::string{});
  node->get_parameter_or("output_dir", output_path, std::string{});
  node->get_parameter_or("group_name", group, std::string("fairino5_v6_group"));
  node->get_parameter_or("export_all_contacts", export_all_contacts, false);
  node->get_parameter_or("distance_request", distance_request, true);
  node->get_parameter_or("bullet_only", bullet_only, false);
  node->get_parameter_or("variant_filter", variant_filter, std::string{});
  node->get_parameter_or("max_contacts", max_contacts_parameter, 4096);
  node->get_parameter_or("max_contacts_per_pair", max_contacts_per_pair_parameter, 64);
  if (max_contacts_parameter <= 0 || max_contacts_per_pair_parameter <= 0)
  {
    RCLCPP_ERROR(node->get_logger(), "max_contacts and max_contacts_per_pair must be positive");
    rclcpp::shutdown();
    return 6;
  }
  const auto max_contacts = static_cast<std::size_t>(max_contacts_parameter);
  const auto max_contacts_per_pair = static_cast<std::size_t>(max_contacts_per_pair_parameter);
  if (candidate_path.empty() || pose_path.empty() || output_path.empty())
  {
    RCLCPP_ERROR(node->get_logger(), "candidate_csv, pose_csv and output_dir are required");
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
  RCLCPP_INFO(node->get_logger(), "reading frozen inputs");
  const auto poses = read_poses(pose_path);
  RCLCPP_INFO(node->get_logger(), "read poses=%zu", poses.size());
  const auto candidates = read_candidates(candidate_path);
  RCLCPP_INFO(node->get_logger(), "read candidates=%zu", candidates.size());
  if (poses.size() != 720 || candidates.empty())
  {
    RCLCPP_ERROR(node->get_logger(), "unexpected inputs: poses=%zu candidates=%zu", poses.size(), candidates.size());
    rclcpp::shutdown();
    return 4;
  }
  const std::vector<Variant> variants = {
    {"case_A_self_collision", true, true, false, false, false, "", "", "self"},
    {"case_B_robot_world", true, true, false, false, true, "", "", "robot_world"},
    {"case_C_official_baseline", true, true, false, false, true, "", "", "full"},
    {"case_D_single_pair_diagnostic", true, true, false, false, true, "shoulder_link", "horseshoe_wall_184", "full"},
    {"no_floor_diagnostic", false, true, false, false, true, "", "", "full"},
    {"unpadded_diagnostic", true, false, false, false, true, "", "", "full"},
    {"acm_disabled_diagnostic", true, true, true, false, true, "", "", "full"},
    {"alternative_bullet_diagnostic", true, true, false, true, true, "", "", "full"},
  };
  std::ofstream contacts(fs::path(output_path) / "native_collision_contacts.csv");
  contacts << "variant,waypoint_id,candidate_id,body_1,body_2,pair_type,contact_x,contact_y,contact_z,normal_x,normal_y,normal_z,penetration_depth,percent_interpolation,raw_contact_count,contact_index,contact_export_status\n";
  std::ofstream summary_file(fs::path(output_path) / "native_collision_variant_summary.csv");
  summary_file << "variant,scope,allowed_body_1,allowed_body_2,raw_candidate_count,colliding_candidate_count,self_colliding_candidate_count,robot_world_colliding_candidate_count,both_collision_candidate_count,valid_node_count,total_contact_count,detector,padded,acm_disabled,include_world,padding_link_count,padding_max,scale_link_count,scale_min,scale_max\n";
  for (const auto& variant : variants)
  {
    if (bullet_only && !variant.bullet)
      continue;
    if (!variant_filter.empty() && variant.name != variant_filter)
      continue;
    RCLCPP_INFO(node->get_logger(), "building variant %s", variant.name.c_str());
    auto scene = build_scene(model, poses, variant.include_floor, variant.bullet, variant.include_world);
    RCLCPP_INFO(node->get_logger(), "built variant %s", variant.name.c_str());
    if (variant.acm_disabled)
    {
      auto& acm = scene->getAllowedCollisionMatrixNonConst();
      acm.setEntry(true);
      for (std::size_t i = 0; i < 240; ++i)
        acm.setEntry("horseshoe_wall_" + std::to_string(i), true);
      acm.setEntry("tunnel_floor", true);
    }
    if (!variant.allowed_body_1.empty() && !variant.allowed_body_2.empty())
      scene->getAllowedCollisionMatrixNonConst().setEntry(variant.allowed_body_1, variant.allowed_body_2, true);
    moveit::core::RobotState state(model);
    const auto* joint_group = model->getJointModelGroup(group);
    if (!joint_group)
    {
      RCLCPP_ERROR(node->get_logger(), "joint group not found: %s", group.c_str());
      rclcpp::shutdown();
      return 5;
    }
    std::size_t colliding = 0, self_colliding = 0, robot_world = 0, both = 0, valid = 0, total_contacts = 0;
    for (const auto& candidate : candidates)
    {
      state.setJointGroupPositions(joint_group, candidate.q);
      state.update();
      const auto result = check_state(*scene, state, group, variant.padded, distance_request,
                                      max_contacts, max_contacts_per_pair);
      total_contacts += result.contact_count;
      colliding += result.full ? 1 : 0;
      self_colliding += result.self ? 1 : 0;
      robot_world += result.robot_world ? 1 : 0;
      both += result.both ? 1 : 0;
      valid += (!result.full) ? 1 : 0;
      std::map<std::pair<std::string, std::string>, const collision_detection::Contact*> representatives;
      for (const auto& contact : result.contacts)
      {
        const auto key = std::make_pair(std::min(contact.body_name_1, contact.body_name_2),
                                        std::max(contact.body_name_1, contact.body_name_2));
        representatives.emplace(key, &contact);
      }
      if (variant.name == "case_C_official_baseline" || variant.name == "case_D_single_pair_diagnostic" ||
          variant.name == "alternative_bullet_diagnostic")
      {
        if (export_all_contacts)
        {
          std::size_t contact_index = 0;
          for (const auto& contact : result.contacts)
            write_contact(contacts, variant.name, candidate, contact, result.contact_count, contact_index++, "all_contacts");
        }
        else
        {
          std::size_t contact_index = 0;
          for (const auto& entry : representatives)
            write_contact(contacts, variant.name, candidate, *entry.second, result.contact_count, contact_index++, "representative_per_body_pair");
        }
      }
    }
    const auto& padding = scene->getCollisionEnv()->getLinkPadding();
    const auto& scale = scene->getCollisionEnv()->getLinkScale();
    double padding_max = 0.0;
    for (const auto& item : padding)
      padding_max = std::max(padding_max, item.second);
    double scale_min = scale.empty() ? 1.0 : scale.begin()->second;
    double scale_max = scale_min;
    for (const auto& item : scale)
    {
      scale_min = std::min(scale_min, item.second);
      scale_max = std::max(scale_max, item.second);
    }
    summary_file << variant.name << ',' << variant.scope << ',' << variant.allowed_body_1 << ',' << variant.allowed_body_2 << ',' << candidates.size() << ',' << colliding << ',' << self_colliding << ',' << robot_world << ',' << both << ',' << valid << ',' << total_contacts << ',' << scene->getCollisionDetectorName() << ',' << (variant.padded ? "true" : "false") << ',' << (variant.acm_disabled ? "true" : "false") << ',' << (variant.include_world ? "true" : "false") << ',' << padding.size() << ',' << std::setprecision(17) << padding_max << ',' << scale.size() << ',' << scale_min << ',' << scale_max << '\n';
  }
  contacts.close();
  summary_file.close();
  rclcpp::shutdown();
  return 0;
}
