#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection/collision_env.hpp>
#include <moveit/collision_detection_fcl/collision_common.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_model/link_model.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <ruckig/ruckig.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <srdfdom/srdfdom/model.h>
#include <urdf/model.h>

namespace fs = std::filesystem;
using moveit::core::RobotState;

namespace {

constexpr std::size_t kDofs = 6;
constexpr double kDefaultJerkBound = 8.0;
constexpr double kNumericalTolerance = 1.0e-10;

struct Args {
  std::string poses, manifest, urdf, srdf, output;
  std::string group{"fairino5_v6_group"};
  double jerk_bound{kDefaultJerkBound};
  double threshold{0.0};
  std::size_t max_depth{12};
  std::size_t stride{1};
  std::string case_filter;
  bool use_native_ruckig_profile{true};
};

struct PoseRow {
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d n{Eigen::Vector3d::UnitZ()};
};

struct Case {
  std::string id, path, family;
};

struct NativeTrajectory {
  std::vector<double> time;
  std::vector<std::array<double, kDofs>> q, v, a;
};

using Coefficients = std::array<double, kDofs>;
using PairDistances = std::map<std::string, double>;

struct StateDistances {
  PairDistances world;
  PairDistances self;
};

struct LinkGeometry {
  std::string name;
  const moveit::core::LinkModel* model{nullptr};
  double shape_radius{0.0};
  Coefficients coefficient{};
};

struct IntervalRecord {
  std::size_t index{0};
  double t0{0.0};
  double t1{0.0};
  double lower_bound{std::numeric_limits<double>::infinity()};
  double static_distance{std::numeric_limits<double>::infinity()};
  double motion_bound{0.0};
  std::string pair;
  std::string domain;
  std::size_t depth{0};
  bool native_ruckig_profile{false};
};

struct CaseResult {
  std::string id;
  std::size_t state_count{0};
  std::size_t interval_count{0};
  std::size_t certified_interval_count{0};
  std::size_t refined_interval_count{0};
  std::size_t max_refinement_depth{0};
  std::size_t static_distance_query_count{0};
  bool backend_ok{true};
  bool certified{false};
  double minimum_certified_clearance{std::numeric_limits<double>::infinity()};
  IntervalRecord worst;
  std::string failure_reason;
};

std::string json_string(const std::string& value)
{
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
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

std::string number(double value)
{
  if (!std::isfinite(value)) return "null";
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

std::string json_array(const std::array<double, kDofs>& values)
{
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << number(values[i]);
  }
  out << ']';
  return out.str();
}

std::vector<std::string> split_csv(const std::string& line)
{
  std::vector<std::string> result;
  std::string item;
  std::stringstream stream(line);
  while (std::getline(stream, item, ',')) {
    if (!item.empty() && item.back() == '\r') item.pop_back();
    if (item.size() >= 2 && item.front() == '"' && item.back() == '"')
      item = item.substr(1, item.size() - 2);
    result.push_back(item);
  }
  if (!line.empty() && line.back() == ',') result.emplace_back();
  return result;
}

std::map<std::string, std::size_t> header_map(const std::vector<std::string>& header)
{
  std::map<std::string, std::size_t> result;
  for (std::size_t i = 0; i < header.size(); ++i) result[header[i]] = i;
  return result;
}

std::size_t required_column(const std::map<std::string, std::size_t>& columns, const std::string& name)
{
  const auto it = columns.find(name);
  if (it == columns.end()) throw std::runtime_error("missing CSV column: " + name);
  return it->second;
}

Args parse_args(int argc, char** argv)
{
  Args args;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key(argv[i]);
    const std::string value(argv[i + 1]);
    if (key == "--poses") args.poses = value;
    else if (key == "--manifest") args.manifest = value;
    else if (key == "--urdf") args.urdf = value;
    else if (key == "--srdf") args.srdf = value;
    else if (key == "--output") args.output = value;
    else if (key == "--group") args.group = value;
    else if (key == "--jerk-bound") args.jerk_bound = std::stod(value);
    else if (key == "--threshold") args.threshold = std::stod(value);
    else if (key == "--max-depth") args.max_depth = static_cast<std::size_t>(std::stoull(value));
    else if (key == "--stride") args.stride = static_cast<std::size_t>(std::stoull(value));
    else if (key == "--case") args.case_filter = value;
    else if (key == "--motion-model") {
      if (value == "native_ruckig_profile") args.use_native_ruckig_profile = true;
      else if (value == "endpoint_jerk_cone") args.use_native_ruckig_profile = false;
      else throw std::runtime_error("motion-model must be native_ruckig_profile or endpoint_jerk_cone");
    }
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (args.poses.empty() || args.manifest.empty() || args.urdf.empty() || args.srdf.empty() || args.output.empty())
    throw std::runtime_error("missing --poses/--manifest/--urdf/--srdf/--output");
  if (!std::isfinite(args.jerk_bound) || args.jerk_bound < 0.0 || !std::isfinite(args.threshold) || args.threshold < 0.0)
    throw std::runtime_error("jerk-bound and threshold must be finite and non-negative");
  if (args.stride == 0) throw std::runtime_error("stride must be positive");
  return args;
}

double to_double(const std::string& value) { return std::stod(value); }

std::vector<PoseRow> read_poses(const fs::path& path)
{
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open poses: " + path.string());
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty poses: " + path.string());
  std::vector<PoseRow> result;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split_csv(line);
    if (fields.size() < 10) throw std::runtime_error("invalid pose row");
    PoseRow row;
    row.p = Eigen::Vector3d(to_double(fields[0]), to_double(fields[1]), to_double(fields[2]));
    row.q = Eigen::Quaterniond(to_double(fields[6]), to_double(fields[3]), to_double(fields[4]), to_double(fields[5]));
    row.q.normalize();
    row.n = Eigen::Vector3d(to_double(fields[7]), to_double(fields[8]), to_double(fields[9]));
    if (row.n.norm() <= kNumericalTolerance) throw std::runtime_error("zero pose normal");
    row.n.normalize();
    result.push_back(row);
  }
  if (result.size() < 2) throw std::runtime_error("fewer than two poses");
  return result;
}

std::vector<Case> read_manifest(const fs::path& path)
{
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open manifest: " + path.string());
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty manifest");
  const auto columns = header_map(split_csv(line));
  const auto id = required_column(columns, "case_id");
  const auto trajectory = required_column(columns, "trajectory_csv");
  const auto family = required_column(columns, "family");
  std::vector<Case> result;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split_csv(line);
    if (fields.size() <= std::max({id, trajectory, family})) throw std::runtime_error("short manifest row");
    result.push_back({fields[id], fields[trajectory], fields[family]});
  }
  if (result.empty()) throw std::runtime_error("manifest contains no cases");
  return result;
}

NativeTrajectory read_native_trajectory(const fs::path& path)
{
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open trajectory: " + path.string());
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty trajectory");
  const auto columns = header_map(split_csv(line));
  const auto time = required_column(columns, "t");
  std::array<std::size_t, kDofs> q_columns{}, v_columns{}, a_columns{};
  for (std::size_t joint = 0; joint < kDofs; ++joint) {
    q_columns[joint] = required_column(columns, "j" + std::to_string(joint + 1) + "_q");
    v_columns[joint] = required_column(columns, "j" + std::to_string(joint + 1) + "_dq");
    a_columns[joint] = required_column(columns, "j" + std::to_string(joint + 1) + "_ddq");
  }
  NativeTrajectory result;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split_csv(line);
    std::size_t largest = time;
    for (std::size_t j = 0; j < kDofs; ++j) largest = std::max({largest, q_columns[j], v_columns[j], a_columns[j]});
    if (fields.size() <= largest) throw std::runtime_error("short native trajectory row");
    const double t = to_double(fields[time]);
    std::array<double, kDofs> q{}, v{}, a{};
    for (std::size_t j = 0; j < kDofs; ++j) {
      q[j] = to_double(fields[q_columns[j]]);
      v[j] = to_double(fields[v_columns[j]]);
      a[j] = to_double(fields[a_columns[j]]);
    }
    if (!std::isfinite(t)) throw std::runtime_error("non-finite trajectory time");
    for (std::size_t j = 0; j < kDofs; ++j)
      if (!std::isfinite(q[j]) || !std::isfinite(v[j]) || !std::isfinite(a[j])) throw std::runtime_error("non-finite trajectory state");
    if (!result.time.empty() && t <= result.time.back()) throw std::runtime_error("non-increasing trajectory time");
    result.time.push_back(t);
    result.q.push_back(q);
    result.v.push_back(v);
    result.a.push_back(a);
  }
  if (result.time.size() < 2) throw std::runtime_error("trajectory has fewer than two states");
  return result;
}

NativeTrajectory stride_trajectory(const NativeTrajectory& source, std::size_t stride)
{
  if (stride <= 1) return source;
  NativeTrajectory result;
  result.time.reserve(source.time.size() / stride + 2);
  for (std::size_t i = 0; i < source.time.size(); i += stride) {
    result.time.push_back(source.time[i]);
    result.q.push_back(source.q[i]);
    result.v.push_back(source.v[i]);
    result.a.push_back(source.a[i]);
  }
  if (result.time.back() != source.time.back()) {
    result.time.push_back(source.time.back());
    result.q.push_back(source.q.back());
    result.v.push_back(source.v.back());
    result.a.push_back(source.a.back());
  }
  return result;
}

moveit_msgs::msg::CollisionObject box(const std::string& id, const Eigen::Vector3d& center,
                                      const Eigen::Vector3d& dimensions, const Eigen::Quaterniond& orientation)
{
  moveit_msgs::msg::CollisionObject object;
  object.header.frame_id = "base_link";
  object.id = id;
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  primitive.dimensions = {dimensions.x(), dimensions.y(), dimensions.z()};
  geometry_msgs::msg::Pose pose;
  pose.position.x = center.x(); pose.position.y = center.y(); pose.position.z = center.z();
  pose.orientation.x = orientation.x(); pose.orientation.y = orientation.y();
  pose.orientation.z = orientation.z(); pose.orientation.w = orientation.w();
  object.primitives.push_back(primitive);
  object.primitive_poses.push_back(pose);
  object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return object;
}

std::unique_ptr<planning_scene::PlanningScene> make_scene(
    const moveit::core::RobotModelConstPtr& model, const std::vector<PoseRow>& poses)
{
  auto scene = std::make_unique<planning_scene::PlanningScene>(model);
  for (std::size_t i = 0; i + 1 < poses.size(); ++i) {
    const Eigen::Vector3d p0 = poses[i].p + 0.260 * poses[i].n;
    const Eigen::Vector3d p1 = poses[i + 1].p + 0.260 * poses[i + 1].n;
    Eigen::Vector3d x = p1 - p0;
    const double length = x.norm();
    if (length <= kNumericalTolerance) continue;
    x /= length;
    Eigen::Vector3d y = Eigen::Vector3d::UnitY();
    Eigen::Vector3d z = x.cross(y);
    if (z.norm() <= kNumericalTolerance) z = poses[i].n;
    else z.normalize();
    y = z.cross(x).normalized();
    Eigen::Matrix3d rotation;
    rotation.col(0) = x; rotation.col(1) = y; rotation.col(2) = z;
    scene->processCollisionObjectMsg(box("horseshoe_wall_" + std::to_string(i), (p0 + p1) / 2.0 + 0.020 * poses[i].n,
                                        Eigen::Vector3d(length + 0.040, 1.10, 0.040), Eigen::Quaterniond(rotation)));
  }
  double xmin = std::numeric_limits<double>::infinity();
  double xmax = -std::numeric_limits<double>::infinity();
  double ymean = 0.0;
  for (const auto& pose : poses) { xmin = std::min(xmin, pose.p.x()); xmax = std::max(xmax, pose.p.x()); ymean += pose.p.y(); }
  ymean /= static_cast<double>(poses.size());
  scene->processCollisionObjectMsg(box("tunnel_floor", Eigen::Vector3d((xmin + xmax) / 2.0, ymean, -0.220),
                                      Eigen::Vector3d(xmax - xmin + 0.040, 1.10, 0.040), Eigen::Quaterniond::Identity()));
  scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  return scene;
}

std::string pair_key(std::string first, std::string second)
{
  if (first > second) std::swap(first, second);
  return first + "|" + second;
}

PairDistances collect_distances(const collision_detection::DistanceResult& result)
{
  PairDistances distances;
  for (const auto& entry : result.distances) {
    for (const auto& value : entry.second) {
      if (!std::isfinite(value.distance) || value.link_names[0].empty() || value.link_names[1].empty()) continue;
      const std::string key = pair_key(value.link_names[0], value.link_names[1]);
      auto it = distances.find(key);
      if (it == distances.end() || value.distance < it->second) distances[key] = value.distance;
    }
  }
  if (distances.empty() && std::isfinite(result.minimum_distance.distance) &&
      !result.minimum_distance.link_names[0].empty() && !result.minimum_distance.link_names[1].empty())
    distances[pair_key(result.minimum_distance.link_names[0], result.minimum_distance.link_names[1])] = result.minimum_distance.distance;
  return distances;
}

collision_detection::DistanceRequest make_distance_request(
    const std::string& group, const moveit::core::RobotModelConstPtr& model,
    const collision_detection::AllowedCollisionMatrix& acm)
{
  collision_detection::DistanceRequest request;
  request.type = collision_detection::DistanceRequestTypes::SINGLE;
  request.group_name = group;
  request.max_contacts_per_body = 4096;
  request.enable_nearest_points = true;
  request.enable_signed_distance = true;
  request.compute_gradient = false;
  request.acm = &acm;
  request.enableGroup(model);
  return request;
}

StateDistances query_state(const collision_detection::CollisionEnv* env, const std::string& group,
                           const moveit::core::RobotModelConstPtr& model,
                           const collision_detection::AllowedCollisionMatrix& acm, const RobotState& state)
{
  const auto request = make_distance_request(group, model, acm);
  collision_detection::DistanceResult world, self;
  world.clear(); self.clear();
  env->distanceRobot(request, world, state);
  env->distanceSelf(request, self, state);
  return {collect_distances(world), collect_distances(self)};
}

double shape_radius(const collision_detection::FCLGeometryConstPtr& geometry)
{
  const auto& aabb = geometry->collision_geometry_->aabb_local;
  double radius = 0.0;
  for (int i = 0; i < 3; ++i) radius = std::max(radius, std::max(std::abs(aabb.min_[i]), std::abs(aabb.max_[i])));
  return std::sqrt(3.0) * radius;
}

std::vector<LinkGeometry> build_link_geometry(const moveit::core::RobotModelConstPtr& model,
                                              const moveit::core::JointModelGroup* group)
{
  std::vector<LinkGeometry> result;
  const std::map<std::string, std::size_t> joint_index = [&]() {
    std::map<std::string, std::size_t> indices;
    const auto& names = group->getActiveJointModelNames();
    if (names.size() != kDofs) throw std::runtime_error("expected six active joints");
    for (std::size_t i = 0; i < names.size(); ++i) indices[names[i]] = i;
    return indices;
  }();
  for (const auto* link : model->getLinkModelsWithCollisionGeometry()) {
    if (!link) continue;
    LinkGeometry item;
    item.name = link->getName();
    item.model = link;
    const auto& origins = link->getCollisionOriginTransforms();
    for (std::size_t i = 0; i < link->getShapes().size(); ++i) {
      const auto geometry = collision_detection::createCollisionGeometry(link->getShapes()[i], link, static_cast<int>(i));
      if (!geometry || !geometry->collision_geometry_) throw std::runtime_error("FCL geometry construction failed: " + item.name);
      const double origin_radius = i < origins.size() ? origins[i].translation().norm() : 0.0;
      item.shape_radius = std::max(item.shape_radius, origin_radius + shape_radius(geometry));
    }
    std::vector<const moveit::core::LinkModel*> chain;
    for (const auto* current = link; current && current->getParentJointModel(); current = current->getParentLinkModel()) chain.push_back(current);
    std::reverse(chain.begin(), chain.end());
    for (const auto& joint : joint_index) {
      std::size_t first = chain.size();
      for (std::size_t index = 0; index < chain.size(); ++index)
        if (chain[index]->getParentJointModel()->getName() == joint.first) { first = index; break; }
      if (first == chain.size()) continue;
      double radius = item.shape_radius;
      for (std::size_t index = first; index < chain.size(); ++index)
        radius += chain[index]->getJointOriginTransform().translation().norm();
      item.coefficient[joint.second] = radius;
    }
    result.push_back(item);
  }
  if (result.empty()) throw std::runtime_error("robot has no collision geometry");
  return result;
}

std::map<std::string, Coefficients> coefficient_map(const std::vector<LinkGeometry>& links)
{
  std::map<std::string, Coefficients> result;
  for (const auto& link : links) result[link.name] = link.coefficient;
  return result;
}

Coefficients pair_coefficient(const std::string& pair, const std::map<std::string, Coefficients>& coefficients,
                              bool self_domain)
{
  const auto separator = pair.find('|');
  if (separator == std::string::npos) throw std::runtime_error("malformed distance pair: " + pair);
  const std::string first = pair.substr(0, separator);
  const std::string second = pair.substr(separator + 1);
  const auto first_it = coefficients.find(first);
  const auto second_it = coefficients.find(second);
  if (self_domain) {
    if (first_it == coefficients.end() || second_it == coefficients.end()) throw std::runtime_error("self pair is not robot geometry: " + pair);
    Coefficients result{};
    for (std::size_t i = 0; i < kDofs; ++i) result[i] = first_it->second[i] + second_it->second[i];
    return result;
  }
  if (first_it != coefficients.end() && second_it == coefficients.end()) return first_it->second;
  if (second_it != coefficients.end() && first_it == coefficients.end()) return second_it->second;
  throw std::runtime_error("world pair does not contain exactly one robot link: " + pair);
}

std::array<double, kDofs> endpoint_motion(const NativeTrajectory& trajectory, std::size_t state, double dt, double jerk_bound)
{
  std::array<double, kDofs> result{};
  for (std::size_t joint = 0; joint < kDofs; ++joint)
    result[joint] = std::abs(trajectory.v[state][joint]) * dt + 0.5 * std::abs(trajectory.a[state][joint]) * dt * dt + jerk_bound * dt * dt * dt / 6.0;
  return result;
}

// A two-sided jerk-cone envelope.  The one-sided Taylor cone is valid but can
// be extremely loose across a long exported waypoint gap.  For each joint we
// intersect it with the cone propagated backward from the other endpoint and
// use the largest value of that intersection.  Both cones follow only from
// the exported q/v/a state and the validated global jerk bound; no sampled
// interior state is treated as a certificate.
std::array<double, kDofs> two_sided_motion(const NativeTrajectory& trajectory,
                                           std::size_t left, std::size_t right,
                                           double jerk_bound, bool from_right = false)
{
  const double dt = trajectory.time[right] - trajectory.time[left];
  std::array<double, kDofs> result{};
  for (std::size_t joint = 0; joint < kDofs; ++joint) {
    const double q_delta = std::abs(trajectory.q[right][joint] - trajectory.q[left][joint]);
    const std::size_t anchor = from_right ? right : left;
    const std::size_t other = from_right ? left : right;
    const double v_left = std::abs(trajectory.v[anchor][joint]);
    const double a_left = std::abs(trajectory.a[anchor][joint]);
    const double v_right = std::abs(trajectory.v[other][joint]);
    const double a_right = std::abs(trajectory.a[other][joint]);
    const auto forward_cone = [&](double time) {
      return v_left * time + 0.5 * a_left * time * time + jerk_bound * time * time * time / 6.0;
    };
    const auto backward_cone = [&](double time) {
      const double remaining = dt - time;
      return q_delta + v_right * remaining + 0.5 * a_right * remaining * remaining +
             jerk_bound * remaining * remaining * remaining / 6.0;
    };
    const auto intersection = [&](double time) { return std::min(forward_cone(time), backward_cone(time)); };
    double maximum = std::max(intersection(0.0), intersection(dt));
    if (forward_cone(dt) > backward_cone(dt)) {
      double low = 0.0, high = dt;
      for (int iteration = 0; iteration < 80; ++iteration) {
        const double middle = 0.5 * (low + high);
        if (forward_cone(middle) <= backward_cone(middle)) low = middle;
        else high = middle;
      }
      maximum = std::max(maximum, intersection(0.5 * (low + high)));
    }
    result[joint] = std::max(0.0, maximum);
  }
  return result;
}

bool native_ruckig_motion(const NativeTrajectory& trajectory, std::size_t left, std::size_t right,
                          std::array<double, kDofs>& from_left, std::array<double, kDofs>& from_right)
{
  const double duration = trajectory.time[right] - trajectory.time[left];
  if (!(duration > 0.0) || !std::isfinite(duration)) return false;
  ruckig::InputParameter<ruckig::DynamicDOFs> input(kDofs);
  input.current_position = std::vector<double>(trajectory.q[left].begin(), trajectory.q[left].end());
  input.target_position = std::vector<double>(trajectory.q[right].begin(), trajectory.q[right].end());
  input.current_velocity = std::vector<double>(trajectory.v[left].begin(), trajectory.v[left].end());
  input.target_velocity = std::vector<double>(trajectory.v[right].begin(), trajectory.v[right].end());
  input.current_acceleration = std::vector<double>(trajectory.a[left].begin(), trajectory.a[left].end());
  input.target_acceleration = std::vector<double>(trajectory.a[right].begin(), trajectory.a[right].end());
  input.max_velocity = {0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48};
  input.max_acceleration = {0.105, 0.105, 0.105, 0.105, 0.105, 0.105};
  input.max_jerk = {8.0, 8.0, 8.0, 8.0, 8.0, 8.0};
  input.minimum_duration = duration;
  ruckig::Ruckig<ruckig::DynamicDOFs> solver(kDofs, 0.01);
  ruckig::Trajectory<ruckig::DynamicDOFs, ruckig::StandardVector> generated(kDofs);
  const auto outcome = solver.calculate(input, generated);
  if (outcome != ruckig::Result::Working && outcome != ruckig::Result::Finished) return false;
  if (std::abs(generated.get_duration() - duration) > 1.0e-9) return false;
  const auto extrema = generated.get_position_extrema();
  for (std::size_t joint = 0; joint < kDofs; ++joint) {
    if (!std::isfinite(extrema[joint].min) || !std::isfinite(extrema[joint].max)) return false;
    from_left[joint] = std::max(std::abs(extrema[joint].min - trajectory.q[left][joint]),
                                std::abs(extrema[joint].max - trajectory.q[left][joint]));
    from_right[joint] = std::max(std::abs(extrema[joint].min - trajectory.q[right][joint]),
                                 std::abs(extrema[joint].max - trajectory.q[right][joint]));
  }
  return true;
}

double dot(const Coefficients& coefficients, const std::array<double, kDofs>& values)
{
  double result = 0.0;
  for (std::size_t i = 0; i < kDofs; ++i) result += coefficients[i] * values[i];
  return result;
}

IntervalRecord interval_bound(std::size_t left_index, std::size_t right_index, const NativeTrajectory& trajectory, const StateDistances& first,
                              const StateDistances& second, const std::map<std::string, Coefficients>& coefficients,
                              double jerk_bound, double threshold, bool use_native_ruckig_profile)
{
  const double t0 = trajectory.time[left_index];
  const double t1 = trajectory.time[right_index];
  const double dt = t1 - t0;
  auto start_motion = two_sided_motion(trajectory, left_index, right_index, jerk_bound);
  auto end_motion = two_sided_motion(trajectory, left_index, right_index, jerk_bound, true);
  std::array<double, kDofs> native_start{}, native_end{};
  const bool native_profile = use_native_ruckig_profile &&
                              native_ruckig_motion(trajectory, left_index, right_index, native_start, native_end);
  if (native_profile) {
    start_motion = native_start;
    end_motion = native_end;
  }
  IntervalRecord best;
  best.index = left_index; best.t0 = t0; best.t1 = t1;
  best.native_ruckig_profile = native_profile;
  const auto inspect = [&](const PairDistances& left, const PairDistances& right, bool self_domain) {
    std::set<std::string> pairs;
    for (const auto& item : left) pairs.insert(item.first);
    for (const auto& item : right) pairs.insert(item.first);
    for (const auto& pair : pairs) {
      const auto left_it = left.find(pair), right_it = right.find(pair);
      if (left_it == left.end() || right_it == right.end()) continue;
      const auto coefficient = pair_coefficient(pair, coefficients, self_domain);
      const double left_bound = left_it->second - dot(coefficient, start_motion);
      const double right_bound = right_it->second - dot(coefficient, end_motion);
      const double lower = std::min(left_bound, right_bound);
      if (lower < best.lower_bound) {
        best.lower_bound = lower;
        best.static_distance = std::min(left_it->second, right_it->second);
        best.motion_bound = best.static_distance - lower;
        best.pair = pair;
        best.domain = self_domain ? "self" : "world";
      }
      (void)threshold;
    }
  };
  inspect(first.world, second.world, false);
  inspect(first.self, second.self, true);
  if (!std::isfinite(best.lower_bound)) {
    best.lower_bound = -std::numeric_limits<double>::infinity();
    best.static_distance = std::numeric_limits<double>::quiet_NaN();
    best.motion_bound = std::numeric_limits<double>::infinity();
    best.domain = "unavailable";
  }
  return best;
}

CaseResult run_case(const Case& item, const Args& args, const std::vector<PoseRow>& poses,
                    const moveit::core::RobotModelConstPtr& model, const moveit::core::JointModelGroup* group,
                    const std::map<std::string, Coefficients>& coefficients)
{
  const NativeTrajectory trajectory = read_native_trajectory(item.path);
  auto scene = make_scene(model, poses);
  const auto& acm = scene->getAllowedCollisionMatrix();
  const auto* env = scene->getCollisionEnv().get();
  std::vector<StateDistances> states;
  states.reserve(trajectory.time.size());
  CaseResult result;
  result.id = item.id;
  result.state_count = trajectory.time.size();
  std::map<std::size_t, StateDistances> distance_cache;
  RobotState state(model);
  const auto query_index = [&](std::size_t index) -> const StateDistances& {
    const auto cached = distance_cache.find(index);
    if (cached != distance_cache.end()) return cached->second;
    state.setJointGroupPositions(group, std::vector<double>(trajectory.q[index].begin(), trajectory.q[index].end()));
    state.update();
    const auto inserted = distance_cache.emplace(index, query_state(env, args.group, model, acm, state));
    return inserted.first->second;
  };
  const auto update_worst = [&](const IntervalRecord& record) {
    if (!std::isfinite(result.minimum_certified_clearance) || record.lower_bound < result.minimum_certified_clearance) {
      result.minimum_certified_clearance = record.lower_bound;
      result.worst = record;
    }
  };
  const auto process_interval = [&](auto&& self, std::size_t left, std::size_t right, std::size_t depth) -> void {
    const auto& first = query_index(left);
    const auto& second = query_index(right);
    IntervalRecord record = interval_bound(left, right, trajectory, first, second, coefficients, args.jerk_bound, args.threshold,
                                           args.use_native_ruckig_profile);
    record.depth = depth;
    result.max_refinement_depth = std::max(result.max_refinement_depth, depth);
    if (record.lower_bound > args.threshold) {
      result.certified_interval_count += right - left;
      update_worst(record);
      return;
    }
    if (right - left > 1 && depth < args.max_depth) {
      const std::size_t middle = left + (right - left) / 2;
      self(self, left, middle, depth + 1);
      self(self, middle, right, depth + 1);
      return;
    }
    result.refined_interval_count += right - left;
    result.certified = false;
    if (!std::isfinite(record.lower_bound)) result.failure_reason = "distance_backend_missing_pair_data";
    else result.failure_reason = "adaptive_endpoint_lipschitz_bound_not_positive";
    update_worst(record);
  };
  result.interval_count = trajectory.time.size() - 1;
  result.certified = true;
  for (std::size_t i = 0; i < trajectory.time.size(); i += args.stride) {
    const std::size_t next = std::min(i + args.stride, trajectory.time.size() - 1);
    process_interval(process_interval, i, next, 0);
    if (next == trajectory.time.size() - 1) break;
  }
  result.static_distance_query_count = distance_cache.size();
  if (result.certified_interval_count == result.interval_count && std::isfinite(result.minimum_certified_clearance)) result.certified = true;
  return result;
}

void write_result(std::ofstream& output, const CaseResult& result, const Case& item, const Args& args)
{
  output << "{\"case_id\":" << json_string(result.id)
         << ",\"family\":" << json_string(item.family)
         << ",\"state_count\":" << result.state_count
         << ",\"interval_count\":" << result.interval_count
         << ",\"certified_interval_count\":" << result.certified_interval_count
         << ",\"refined_interval_count\":" << result.refined_interval_count
         << ",\"max_refinement_depth\":" << result.max_refinement_depth
         << ",\"static_distance_query_count\":" << result.static_distance_query_count
         << ",\"certified\":" << (result.certified ? "true" : "false")
         << ",\"minimum_certified_clearance_m\":" << number(result.minimum_certified_clearance)
         << ",\"worst_case\":{\"domain\":" << json_string(result.worst.domain)
         << ",\"pair\":" << json_string(result.worst.pair)
         << ",\"time_start_s\":" << number(result.worst.t0)
         << ",\"time_end_s\":" << number(result.worst.t1)
         << ",\"static_distance_m\":" << number(result.worst.static_distance)
         << ",\"motion_bound_m\":" << number(result.worst.motion_bound)
         << ",\"lower_bound_m\":" << number(result.worst.lower_bound)
         << ",\"refinement_depth\":" << result.worst.depth
         << ",\"native_ruckig_profile\":" << (result.worst.native_ruckig_profile ? "true" : "false") << "}"
         << ",\"threshold_m\":" << number(args.threshold)
         << ",\"jerk_bound_rad_s3\":" << number(args.jerk_bound)
         << ",\"trajectory_stride\":" << args.stride
         << ",\"backend\":\"MoveIt2 CollisionEnvFCL DistanceRequest SINGLE\""
         << ",\"collision_method\":\"adaptive_discrete_interpolation\""
         << ",\"status\":" << json_string(result.certified ? "PASS_CERTIFIED_MODEL_SPACE_CLEARANCE" : "UNRESOLVED_NOT_CERTIFIED")
         << ",\"failure_reason\":" << json_string(result.failure_reason) << "}\n";
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    const Args args = parse_args(argc, argv);
    fs::create_directories(args.output);
    const auto poses = read_poses(args.poses);
    const auto cases = read_manifest(args.manifest);
    rclcpp::init(argc, argv);
    auto urdf = std::make_shared<urdf::Model>();
    if (!urdf->initFile(args.urdf)) throw std::runtime_error("URDF parse failed");
    auto srdf = std::make_shared<srdf::Model>();
    if (!srdf->initFile(*urdf, args.srdf)) throw std::runtime_error("SRDF parse failed");
    auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
    const auto* group = model->getJointModelGroup(args.group);
    if (!group) throw std::runtime_error("planning group missing: " + args.group);
    if (group->getActiveJointModelNames() != std::vector<std::string>{"j1", "j2", "j3", "j4", "j5", "j6"})
      throw std::runtime_error("active joint order mismatch");
    const auto links = build_link_geometry(model, group);
    const auto coefficients = coefficient_map(links);
    std::ofstream summaries(fs::path(args.output) / "continuous_clearance_case_summary.jsonl");
    if (!summaries) throw std::runtime_error("cannot open case summary output");
    std::size_t pass_count = 0;
    double global_min = std::numeric_limits<double>::infinity();
    CaseResult global_worst;
    std::vector<std::string> failed;
    std::size_t selected_case_count = 0;
    for (const auto& item : cases) {
      if (!args.case_filter.empty() && item.id != args.case_filter) continue;
      ++selected_case_count;
      const auto result = run_case(item, args, poses, model, group, coefficients);
      write_result(summaries, result, item, args);
      if (result.certified) { ++pass_count; if (result.minimum_certified_clearance < global_min) { global_min = result.minimum_certified_clearance; global_worst = result; } }
      else { failed.push_back(item.id); if (!std::isfinite(global_min) || result.minimum_certified_clearance < global_min) { global_min = result.minimum_certified_clearance; global_worst = result; } }
      std::cerr << "D51 clearance case=" << item.id << " status=" << (result.certified ? "PASS" : "UNRESOLVED")
                << " min_lb=" << number(result.minimum_certified_clearance) << " intervals=" << result.interval_count << "\n";
    }
    std::ofstream report(fs::path(args.output) / "continuous_clearance_summary.json");
    if (selected_case_count == 0) throw std::runtime_error("case filter matched no manifest row");
    report << "{\n"
           << "  \"schema_version\":\"d51-continuous-clearance-v1\",\n"
           << "  \"status\":" << json_string(pass_count == selected_case_count ? "PASS" : "UNRESOLVED_NOT_CERTIFIED") << ",\n"
           << "  \"backend\":\"MoveIt2 CollisionEnvFCL DistanceRequest SINGLE\",\n"
           << "  \"method\":" << json_string(args.use_native_ruckig_profile
                                                    ? "adaptive_pair_specific_interval_bound_with_two_sided_jerk_cone_and_duration_consistent_native_ruckig_extrema"
                                                    : "adaptive_pair_specific_interval_bound_with_two_sided_endpoint_jerk_cone") << ",\n"
           << "  \"motion_model\":" << json_string(args.use_native_ruckig_profile ? "native_ruckig_profile" : "endpoint_jerk_cone") << ",\n"
           << "  \"collision_method\":\"adaptive_discrete_interpolation\",\n"
           << "  \"threshold_m\":" << number(args.threshold) << ",\n"
           << "  \"jerk_bound_rad_s3\":" << number(args.jerk_bound) << ",\n"
           << "  \"case_count\":" << selected_case_count << ",\n"
           << "  \"certified_case_count\":" << pass_count << ",\n"
           << "  \"minimum_certified_clearance_m\":" << number(global_min) << ",\n"
           << "  \"worst_case_id\":" << json_string(global_worst.id) << ",\n"
           << "  \"worst_pair\":" << json_string(global_worst.worst.pair) << ",\n"
           << "  \"worst_interval_start_s\":" << number(global_worst.worst.t0) << ",\n"
           << "  \"worst_interval_end_s\":" << number(global_worst.worst.t1) << ",\n"
           << "  \"worst_static_distance_m\":" << number(global_worst.worst.static_distance) << ",\n"
           << "  \"worst_motion_bound_m\":" << number(global_worst.worst.motion_bound) << ",\n"
           << "  \"worst_lower_bound_m\":" << number(global_worst.worst.lower_bound) << ",\n"
           << "  \"worst_native_ruckig_profile\":" << (global_worst.worst.native_ruckig_profile ? "true" : "false") << ",\n"
           << "  \"trajectory_stride\":" << args.stride << ",\n"
           << "  \"failed_case_ids\":[";
    for (std::size_t i = 0; i < failed.size(); ++i) { if (i) report << ','; report << json_string(failed[i]); }
    report << "],\n  \"status_semantics\":\"positive lower bound is model-space certified only; hardware clearance remains unavailable\"\n}\n";
    summaries.close();
    report.close();
    rclcpp::shutdown();
    return pass_count == selected_case_count ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "stage4e_continuous_clearance ERROR: " << error.what() << "\n";
    if (rclcpp::ok()) rclcpp::shutdown();
    return 1;
  }
}
