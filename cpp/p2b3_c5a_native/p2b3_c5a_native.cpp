#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection/collision_env.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <srdfdom/srdfdom/model.h>
#include <urdf/model.h>

namespace fs = std::filesystem;
using moveit::core::RobotState;

namespace {
constexpr std::size_t kDofs = 6;
constexpr double kMaxStep = 0.008726646259971648;  // 0.5 degrees, C4 frozen policy.

struct Args {
  std::string cases, urdf, srdf, output;
  std::string group{"fairino5_v6_group"};
  std::string tip{"spray_tcp_link"};
};

struct JointRow { double time{0.0}; std::array<double, kDofs> q{}; };
struct Case { std::string id, q_path, scene_path; };
struct SceneObject {
  std::string id;
  Eigen::Vector3d dimensions{Eigen::Vector3d::Zero()};
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};
struct SampleContext {
  std::size_t sample{0};
  int waypoint{-1};
  int segment_start{0};
  int segment_end{0};
  double fraction{0.0};
  double time{0.0};
};
struct Minimum {
  bool available{false};
  double distance{std::numeric_limits<double>::quiet_NaN()};
  std::string names[2];
  Eigen::Vector3d points[2]{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  SampleContext context;
};

std::string json_quote(const std::string& value) {
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
std::string number(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream out; out << std::setprecision(17) << value; return out.str();
}
std::string vector_json(const Eigen::Vector3d& value) {
  return "[" + number(value.x()) + "," + number(value.y()) + "," + number(value.z()) + "]";
}
std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> fields; std::stringstream stream(line); std::string item;
  while (std::getline(stream, item, ',')) { if (!item.empty() && item.back() == '\r') item.pop_back(); fields.push_back(item); }
  if (!line.empty() && line.back() == ',') fields.emplace_back();
  return fields;
}
std::vector<std::string> header_map(const std::string& line) { return split_csv(line); }
std::size_t field_index(const std::vector<std::string>& header, const std::string& name) {
  const auto it = std::find(header.begin(), header.end(), name);
  if (it == header.end()) throw std::runtime_error("csv_field_missing:" + name);
  return static_cast<std::size_t>(it - header.begin());
}
double parse_number(const std::string& value, const std::string& label) {
  const double parsed = std::stod(value);
  if (!std::isfinite(parsed)) throw std::runtime_error("nonfinite_csv_value:" + label);
  return parsed;
}
Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key(argv[i]), value(argv[i + 1]);
    if (key == "--cases") args.cases = value;
    else if (key == "--urdf") args.urdf = value;
    else if (key == "--srdf") args.srdf = value;
    else if (key == "--output") args.output = value;
    else if (key == "--group") args.group = value;
    else if (key == "--tip") args.tip = value;
    else throw std::runtime_error("unknown_argument:" + key);
  }
  if (args.cases.empty() || args.urdf.empty() || args.srdf.empty() || args.output.empty())
    throw std::runtime_error("required_arguments:--cases --urdf --srdf --output");
  return args;
}
std::vector<Case> read_cases(const fs::path& path) {
  std::ifstream input(path); if (!input) throw std::runtime_error("cannot_open_case_manifest:" + path.string());
  std::string line; if (!std::getline(input, line)) throw std::runtime_error("empty_case_manifest");
  const auto header = header_map(line); const auto id_i = field_index(header, "case_id");
  const auto q_i = field_index(header, "q_csv"); const auto scene_i = field_index(header, "scene_csv");
  std::vector<Case> result;
  while (std::getline(input, line)) {
    if (line.empty()) continue; const auto fields = split_csv(line);
    if (fields.size() != header.size()) throw std::runtime_error("case_manifest_column_count_mismatch");
    result.push_back({fields[id_i], fields[q_i], fields[scene_i]});
  }
  if (result.empty()) throw std::runtime_error("case_manifest_has_no_cases");
  return result;
}
std::vector<JointRow> read_joint_csv(const fs::path& path) {
  std::ifstream input(path); if (!input) throw std::runtime_error("cannot_open_q_csv:" + path.string());
  std::string line; if (!std::getline(input, line)) throw std::runtime_error("empty_q_csv");
  const auto header = header_map(line); const auto time_i = field_index(header, "t");
  std::array<std::size_t, kDofs> q_i{};
  for (std::size_t j = 0; j < kDofs; ++j) q_i[j] = field_index(header, "j" + std::to_string(j + 1) + "_q");
  std::vector<JointRow> rows;
  while (std::getline(input, line)) {
    if (line.empty()) continue; const auto fields = split_csv(line);
    if (fields.size() != header.size()) throw std::runtime_error("q_csv_column_count_mismatch");
    JointRow row; row.time = parse_number(fields[time_i], "t");
    for (std::size_t j = 0; j < kDofs; ++j) row.q[j] = parse_number(fields[q_i[j]], "j_q");
    rows.push_back(row);
  }
  if (rows.size() != 181) throw std::runtime_error("q_waypoint_count_must_be_181");
  for (std::size_t i = 1; i < rows.size(); ++i)
    if (!(rows[i].time > rows[i - 1].time)) throw std::runtime_error("q_timestamps_not_strictly_increasing");
  return rows;
}
std::vector<SceneObject> read_scene_csv(const fs::path& path) {
  std::ifstream input(path); if (!input) throw std::runtime_error("cannot_open_scene_csv:" + path.string());
  std::string line; if (!std::getline(input, line)) throw std::runtime_error("empty_scene_csv");
  const auto header = header_map(line);
  const auto id_i = field_index(header, "id"), dx_i = field_index(header, "dx"), dy_i = field_index(header, "dy"), dz_i = field_index(header, "dz");
  const auto px_i = field_index(header, "px"), py_i = field_index(header, "py"), pz_i = field_index(header, "pz");
  const auto qx_i = field_index(header, "qx"), qy_i = field_index(header, "qy"), qz_i = field_index(header, "qz"), qw_i = field_index(header, "qw");
  std::vector<SceneObject> result;
  while (std::getline(input, line)) {
    if (line.empty()) continue; const auto fields = split_csv(line);
    if (fields.size() != header.size()) throw std::runtime_error("scene_csv_column_count_mismatch");
    SceneObject o; o.id = fields[id_i];
    o.dimensions = {parse_number(fields[dx_i], "dx"), parse_number(fields[dy_i], "dy"), parse_number(fields[dz_i], "dz")};
    o.position = {parse_number(fields[px_i], "px"), parse_number(fields[py_i], "py"), parse_number(fields[pz_i], "pz")};
    o.orientation = Eigen::Quaterniond(parse_number(fields[qw_i], "qw"), parse_number(fields[qx_i], "qx"), parse_number(fields[qy_i], "qy"), parse_number(fields[qz_i], "qz"));
    if (o.dimensions.minCoeff() <= 0.0 || o.orientation.norm() < 1.0e-12) throw std::runtime_error("invalid_scene_box:" + o.id);
    o.orientation.normalize(); result.push_back(std::move(o));
  }
  if (result.size() != 181) throw std::runtime_error("scene_object_count_must_be_181");
  return result;
}
moveit_msgs::msg::CollisionObject box_message(const SceneObject& object) {
  moveit_msgs::msg::CollisionObject message; message.header.frame_id = "base_link"; message.id = object.id;
  shape_msgs::msg::SolidPrimitive primitive; primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  primitive.dimensions = {object.dimensions.x(), object.dimensions.y(), object.dimensions.z()};
  geometry_msgs::msg::Pose pose; pose.position.x = object.position.x(); pose.position.y = object.position.y(); pose.position.z = object.position.z();
  pose.orientation.x = object.orientation.x(); pose.orientation.y = object.orientation.y(); pose.orientation.z = object.orientation.z(); pose.orientation.w = object.orientation.w();
  message.primitives.push_back(primitive); message.primitive_poses.push_back(pose); message.operation = moveit_msgs::msg::CollisionObject::ADD;
  return message;
}
std::unique_ptr<planning_scene::PlanningScene> make_scene(const moveit::core::RobotModelConstPtr& model, const fs::path& path) {
  auto scene = std::make_unique<planning_scene::PlanningScene>(model);
  const auto objects = read_scene_csv(path);
  for (const auto& object : objects)
    if (!scene->processCollisionObjectMsg(box_message(object))) throw std::runtime_error("scene_rejected_object:" + object.id);
  if (scene->getWorld()->getObjectIds().size() != 181) throw std::runtime_error("installed_scene_object_count_mismatch");
  scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  return scene;
}
collision_detection::CollisionRequest collision_request(const std::string& group) {
  collision_detection::CollisionRequest request; request.group_name = group; request.contacts = true;
  request.max_contacts = 64; request.max_contacts_per_pair = 2;
  request.pad_environment_collisions = true; request.pad_self_collisions = false;
  return request;
}
collision_detection::DistanceRequest distance_request(const std::string& group,
                                                       const moveit::core::RobotModelConstPtr& model,
                                                       const collision_detection::AllowedCollisionMatrix& acm) {
  collision_detection::DistanceRequest request; request.type = collision_detection::DistanceRequestTypes::SINGLE;
  request.max_contacts_per_body = 4096;
  request.group_name = group; request.enable_nearest_points = true; request.enable_signed_distance = true;
  request.acm = &acm; request.enableGroup(model); return request;
}
bool update_minimum(Minimum& minimum, const collision_detection::DistanceResult& result, const SampleContext& context) {
  const double sentinel = std::numeric_limits<double>::max() * 0.5;
  bool sample_has_pair = false;
  const auto consider = [&](const collision_detection::DistanceResultsData& data) {
    if (!std::isfinite(data.distance) || std::abs(data.distance) >= sentinel || data.link_names[0].empty() ||
        data.link_names[1].empty() || !data.nearest_points[0].allFinite() || !data.nearest_points[1].allFinite()) return;
    sample_has_pair = true;
    if (minimum.available && data.distance >= minimum.distance) return;
    minimum.available = true; minimum.distance = data.distance; minimum.context = context;
    for (int i = 0; i < 2; ++i) { minimum.names[i] = data.link_names[i]; minimum.points[i] = data.nearest_points[i]; }
  };
  consider(result.minimum_distance);
  for (const auto& pair : result.distances)
    for (const auto& value : pair.second) consider(value);
  return sample_has_pair;
}
void set_state(RobotState& state, const moveit::core::JointModelGroup* group, const std::array<double, kDofs>& q) {
  state.setJointGroupPositions(group, std::vector<double>(q.begin(), q.end())); state.update();
}
std::string contact_pairs(const collision_detection::CollisionResult& result) {
  std::ostringstream out; out << '['; bool first = true;
  for (const auto& entry : result.contacts) {
    if (!first) out << ','; first = false;
    out << '[' << json_quote(entry.first.first) << ',' << json_quote(entry.first.second) << ']';
  }
  out << ']'; return out.str();
}
void write_minimum(std::ostream& out, const Minimum& minimum) {
  if (!minimum.available) { out << "{\"status\":\"NOT_AVAILABLE_NO_FINITE_PAIR\",\"distance_m\":null,\"pair\":null,\"nearest_points\":null}"; return; }
  out << "{\"status\":\"AVAILABLE\",\"distance_m\":" << number(minimum.distance)
      << ",\"pair\":[" << json_quote(minimum.names[0]) << ',' << json_quote(minimum.names[1]) << ']'
      << ",\"nearest_points\":[" << vector_json(minimum.points[0]) << ',' << vector_json(minimum.points[1]) << ']'
      << ",\"sample_index\":" << minimum.context.sample << ",\"waypoint_index\":" << minimum.context.waypoint
      << ",\"segment\":\"" << minimum.context.segment_start << "->" << minimum.context.segment_end
      << "\",\"segment_fraction\":" << number(minimum.context.fraction) << ",\"time_s\":" << number(minimum.context.time) << '}';
}
std::vector<SampleContext> sample_contexts(const std::vector<JointRow>& rows, std::vector<std::array<double, kDofs>>& q_samples) {
  std::vector<SampleContext> contexts; contexts.reserve(900);
  q_samples.reserve(900); q_samples.push_back(rows.front().q);
  contexts.push_back({0, 0, 0, 0, 0.0, rows.front().time});
  for (std::size_t segment = 0; segment + 1 < rows.size(); ++segment) {
    double max_delta = 0.0;
    for (std::size_t j = 0; j < kDofs; ++j) max_delta = std::max(max_delta, std::abs(rows[segment + 1].q[j] - rows[segment].q[j]));
    const int steps = std::max(1, static_cast<int>(std::ceil(max_delta / kMaxStep)));
    for (int k = 1; k <= steps; ++k) {
      const double fraction = static_cast<double>(k) / steps; std::array<double, kDofs> q{};
      for (std::size_t j = 0; j < kDofs; ++j) q[j] = rows[segment].q[j] + fraction * (rows[segment + 1].q[j] - rows[segment].q[j]);
      q_samples.push_back(q);
      contexts.push_back({contexts.size(), k == steps ? static_cast<int>(segment + 1) : -1,
                          static_cast<int>(segment), static_cast<int>(segment + 1), fraction,
                          rows[segment].time + fraction * (rows[segment + 1].time - rows[segment].time)});
    }
  }
  return contexts;
}
void write_fk(const fs::path& path, const std::string& case_id, const std::vector<JointRow>& rows,
              const moveit::core::RobotModelConstPtr& model, const moveit::core::JointModelGroup* group,
              const moveit::core::LinkModel* tip, bool write_header) {
  std::ofstream out(path, write_header ? std::ios::trunc : std::ios::app);
  if (!out) throw std::runtime_error("cannot_write_fk_trace");
  if (write_header) out << "case_id,waypoint,time_s,px,py,pz,r00,r01,r02,r10,r11,r12,r20,r21,r22\n";
  RobotState state(model);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    set_state(state, group, rows[i].q); const Eigen::Isometry3d& transform = state.getGlobalLinkTransform(tip);
    out << case_id << ',' << i << ',' << number(rows[i].time) << ',' << number(transform.translation().x()) << ','
        << number(transform.translation().y()) << ',' << number(transform.translation().z());
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) out << ',' << number(transform.linear()(r, c));
    out << '\n';
  }
}
void run_case(const Case& c, const fs::path& output, const moveit::core::RobotModelConstPtr& model,
              const std::string& group_name, const std::string& tip_name, bool first_fk) {
  const auto rows = read_joint_csv(c.q_path); auto scene = make_scene(model, c.scene_path);
  const auto* group = model->getJointModelGroup(group_name); const auto* tip = model->getLinkModel(tip_name);
  if (!group || !tip) throw std::runtime_error("group_or_tip_missing");
  std::vector<std::string> active(group->getActiveJointModelNames().begin(), group->getActiveJointModelNames().end());
  const std::vector<std::string> expected{"j1", "j2", "j3", "j4", "j5", "j6"};
  if (active != expected) throw std::runtime_error("active_joint_order_mismatch");
  const auto& acm = scene->getAllowedCollisionMatrix(); const auto* env = scene->getCollisionEnv().get();
  const auto* distance_env = scene->getCollisionEnvUnpadded().get();
  if (!env || !distance_env) throw std::runtime_error("fcl_environment_unavailable");
  const auto creq = collision_request(group_name); const auto dreq = distance_request(group_name, model, acm);
  std::vector<std::array<double, kDofs>> q_samples; const auto contexts = sample_contexts(rows, q_samples);
  int world_collision_count = 0, self_collision_count = 0;
  std::size_t world_distance_valid_count = 0, self_distance_valid_count = 0;
  Minimum min_world, min_self; std::string first_world_collision = "null", first_self_collision = "null";
  RobotState state(model);
  for (std::size_t i = 0; i < q_samples.size(); ++i) {
    set_state(state, group, q_samples[i]);
    collision_detection::CollisionResult world_collision, self_collision;
    env->checkRobotCollision(creq, world_collision, state, acm);
    env->checkSelfCollision(creq, self_collision, state, acm);
    collision_detection::DistanceResult world_distance, self_distance;
    world_distance.clear(); self_distance.clear(); distance_env->distanceRobot(dreq, world_distance, state); distance_env->distanceSelf(dreq, self_distance, state);
    world_distance_valid_count += static_cast<std::size_t>(update_minimum(min_world, world_distance, contexts[i]));
    self_distance_valid_count += static_cast<std::size_t>(update_minimum(min_self, self_distance, contexts[i]));
    if (world_collision.collision) {
      ++world_collision_count;
      if (first_world_collision == "null") { std::ostringstream record; record << "{\"sample_index\":" << i << ",\"waypoint_index\":" << contexts[i].waypoint << ",\"segment_start_waypoint\":" << contexts[i].segment_start << ",\"segment_end_waypoint\":" << contexts[i].segment_end << ",\"segment_fraction\":" << number(contexts[i].fraction) << ",\"time_s\":" << number(contexts[i].time) << ",\"pairs\":" << contact_pairs(world_collision) << '}'; first_world_collision = record.str(); }
    }
    if (self_collision.collision) {
      ++self_collision_count;
      if (first_self_collision == "null") { std::ostringstream record; record << "{\"sample_index\":" << i << ",\"waypoint_index\":" << contexts[i].waypoint << ",\"segment_start_waypoint\":" << contexts[i].segment_start << ",\"segment_end_waypoint\":" << contexts[i].segment_end << ",\"segment_fraction\":" << number(contexts[i].fraction) << ",\"time_s\":" << number(contexts[i].time) << ",\"pairs\":" << contact_pairs(self_collision) << '}'; first_self_collision = record.str(); }
    }
  }
  write_fk(output / "p2b3_c5a_fresh_fk.csv", c.id, rows, model, group, tip, first_fk);
  std::ofstream summary(output / "p2b3_c5a_native_cases.jsonl", first_fk ? std::ios::trunc : std::ios::app);
  if (!summary) throw std::runtime_error("cannot_write_native_case_summary");
  summary << "{\"case_id\":" << json_quote(c.id) << ",\"sample_count\":" << q_samples.size()
          << ",\"collision_method\":\"adaptive_discrete_interpolation\",\"max_joint_step_rad\":" << number(kMaxStep)
          << ",\"collision_backend\":\"MoveIt2 PlanningScene active FCL environment\",\"collision_padding_flags\":{\"environment\":true,\"self\":false}"
          << ",\"distance_backend\":\"MoveIt2 CollisionEnvFCL getCollisionEnvUnpadded distanceRobot/distanceSelf\",\"distance_request_type\":\"SINGLE_per_pair_minimum_reduced_to_global_minimum\",\"distance_padding\":0.0,\"distance_scale\":1.0"
          << ",\"robot_world_collision_samples\":" << world_collision_count << ",\"self_collision_samples\":" << self_collision_count
          << ",\"robot_world_distance_valid_samples\":" << world_distance_valid_count
          << ",\"self_distance_valid_samples\":" << self_distance_valid_count
          << ",\"first_robot_world_collision\":" << first_world_collision << ",\"first_self_collision\":" << first_self_collision
          << ",\"minimum_robot_world_clearance\":"; write_minimum(summary, min_world);
  summary << ",\"minimum_self_clearance\":"; write_minimum(summary, min_self); summary << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv); fs::create_directories(args.output);
    if (!rclcpp::ok()) rclcpp::init(argc, argv);
    auto urdf = std::make_shared<urdf::Model>(); if (!urdf->initFile(args.urdf)) throw std::runtime_error("URDF_parse_failed");
    auto srdf = std::make_shared<srdf::Model>(); if (!srdf->initFile(*urdf, args.srdf)) throw std::runtime_error("SRDF_parse_failed");
    auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
    if (!model->hasJointModelGroup(args.group)) throw std::runtime_error("robot_group_missing:" + args.group);
    const auto cases = read_cases(args.cases);
    for (std::size_t i = 0; i < cases.size(); ++i) {
      run_case(cases[i], args.output, model, args.group, args.tip, i == 0);
      std::cout << "completed_case=" << cases[i].id << " (" << (i + 1) << '/' << cases.size() << ")\n" << std::flush;
    }
    rclcpp::shutdown(); return 0;
  } catch (const std::exception& error) {
    std::cerr << "p2b3_c5a_native: " << error.what() << '\n';
    if (rclcpp::ok()) rclcpp::shutdown(); return 2;
  }
}
