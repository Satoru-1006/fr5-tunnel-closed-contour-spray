#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcl/narrowphase/distance.h>
#include <moveit/collision_detection/collision_matrix.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <moveit/collision_detection_fcl/collision_env_fcl.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_model/joint_model.hpp>
#include <moveit/robot_model/revolute_joint_model.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <srdfdom/srdfdom/model.h>
#include <urdf/model.h>

namespace fs = std::filesystem;
using moveit::core::RobotState;

namespace {

constexpr std::size_t kDofs = 6;
constexpr double kNumericalTolerance = 1.0e-9;

struct Args {
  std::string manifest;
  std::string urdf;
  std::string srdf;
  std::string output;
  std::string group{"fairino5_v6_group"};
  std::size_t initial_stride{16};
  std::size_t max_depth{12};
  double jerk_bound{8.0};
  double threshold{0.0};
  std::string case_filter;
};

struct Case {
  std::string id;
  std::string trajectory;
  std::string family;
};

struct NativeTrajectory {
  std::vector<double> time;
  std::vector<std::array<double, kDofs>> q;
  std::vector<std::array<double, kDofs>> velocity;
  std::vector<std::array<double, kDofs>> acceleration;
  std::vector<std::array<double, kDofs>> jerk;
};

struct ShapeRef {
  std::string link;
  std::size_t shape_index{0};
  const moveit::core::LinkModel* link_model{nullptr};
  collision_detection::FCLGeometryConstPtr geometry;
  double local_radius{0.0};
};

struct LinkInfo {
  std::string name;
  const moveit::core::LinkModel* model{nullptr};
  std::vector<std::size_t> shapes;
  std::array<double, kDofs> motion_coefficient{};
};

struct PairInfo {
  std::string key;
  std::size_t first_link{0};
  std::size_t second_link{0};
  bool allowed{false};
  std::vector<std::pair<std::size_t, std::size_t>> shape_pairs;
  std::array<double, kDofs> motion_coefficient{};
};

struct StateDistances {
  std::vector<double> pair_distance;
  bool finite{true};
};

using StateDistanceCache = std::unordered_map<std::string, StateDistances>;

enum class Status { CertifiedClear, CollisionFound, Unresolved };

struct IntervalEvaluation {
  Status status{Status::Unresolved};
  double lower_bound{std::numeric_limits<double>::quiet_NaN()};
  double start_distance{std::numeric_limits<double>::quiet_NaN()};
  double end_distance{std::numeric_limits<double>::quiet_NaN()};
  double motion_bound{std::numeric_limits<double>::quiet_NaN()};
};

struct CaseResult {
  std::string id;
  std::string family;
  std::size_t state_count{0};
  std::size_t interval_count{0};
  std::size_t max_refinement_depth{0};
  std::vector<std::vector<Status>> statuses;
  std::vector<std::vector<double>> lower_bounds;
  std::vector<std::vector<double>> start_distances;
  std::vector<std::vector<double>> end_distances;
  std::vector<std::vector<double>> motion_bounds;
  std::size_t certified_region_count{0};
  std::size_t collision_region_count{0};
  std::size_t unresolved_region_count{0};
  double minimum_certified_clearance{std::numeric_limits<double>::infinity()};
  double minimum_observed_endpoint_distance{std::numeric_limits<double>::infinity()};
  std::string worst_pair;
  std::size_t worst_interval{0};
  double worst_t0{std::numeric_limits<double>::quiet_NaN()};
  double worst_t1{std::numeric_limits<double>::quiet_NaN()};
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

std::string status_string(Status status)
{
  if (status == Status::CertifiedClear) return "CERTIFIED_CLEAR";
  if (status == Status::CollisionFound) return "COLLISION_FOUND";
  return "UNRESOLVED";
}

std::vector<std::string> split_csv(const std::string& line)
{
  std::vector<std::string> result;
  std::string item;
  std::stringstream stream(line);
  while (std::getline(stream, item, ',')) {
    if (!item.empty() && item.back() == '\r') item.pop_back();
    result.push_back(item);
  }
  if (!line.empty() && line.back() == ',') result.emplace_back();
  return result;
}

std::map<std::string, std::size_t> header_map(const std::vector<std::string>& header)
{
  std::map<std::string, std::size_t> result;
  for (std::size_t index = 0; index < header.size(); ++index) result[header[index]] = index;
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
  for (int index = 1; index + 1 < argc; index += 2) {
    const std::string key(argv[index]);
    const std::string value(argv[index + 1]);
    if (key == "--manifest") args.manifest = value;
    else if (key == "--urdf") args.urdf = value;
    else if (key == "--srdf") args.srdf = value;
    else if (key == "--output") args.output = value;
    else if (key == "--group") args.group = value;
    else if (key == "--initial-stride") args.initial_stride = std::stoull(value);
    else if (key == "--max-depth") args.max_depth = std::stoull(value);
    else if (key == "--jerk-bound") args.jerk_bound = std::stod(value);
    else if (key == "--threshold") args.threshold = std::stod(value);
    else if (key == "--case") args.case_filter = value;
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (args.manifest.empty() || args.urdf.empty() || args.srdf.empty() || args.output.empty())
    throw std::runtime_error("missing --manifest/--urdf/--srdf/--output");
  if (args.initial_stride == 0 || args.jerk_bound < 0.0 || !std::isfinite(args.jerk_bound))
    throw std::runtime_error("invalid interval or jerk configuration");
  return args;
}

std::vector<Case> read_manifest(const fs::path& path)
{
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open manifest: " + path.string());
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty manifest: " + path.string());
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
  if (!std::getline(input, line)) throw std::runtime_error("empty trajectory: " + path.string());
  const auto columns = header_map(split_csv(line));
  const auto time = required_column(columns, "t");
  std::array<std::size_t, kDofs> q_columns{}, v_columns{}, a_columns{}, j_columns{};
  for (std::size_t joint = 0; joint < kDofs; ++joint) {
    const auto suffix = std::to_string(joint + 1);
    q_columns[joint] = required_column(columns, "j" + suffix + "_q");
    v_columns[joint] = required_column(columns, "j" + suffix + "_dq");
    a_columns[joint] = required_column(columns, "j" + suffix + "_ddq");
    j_columns[joint] = required_column(columns, "j" + suffix + "_jerk");
  }
  NativeTrajectory result;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split_csv(line);
    const auto max_column = std::max({time, *std::max_element(q_columns.begin(), q_columns.end()),
                                      *std::max_element(v_columns.begin(), v_columns.end()),
                                      *std::max_element(a_columns.begin(), a_columns.end()),
                                      *std::max_element(j_columns.begin(), j_columns.end())});
    if (fields.size() <= max_column) throw std::runtime_error("short native trajectory row");
    const double t = std::stod(fields[time]);
    std::array<double, kDofs> q{}, velocity{}, acceleration{}, jerk{};
    for (std::size_t joint = 0; joint < kDofs; ++joint) {
      q[joint] = std::stod(fields[q_columns[joint]]);
      velocity[joint] = std::stod(fields[v_columns[joint]]);
      acceleration[joint] = std::stod(fields[a_columns[joint]]);
      jerk[joint] = std::stod(fields[j_columns[joint]]);
    }
    const bool finite = std::isfinite(t) && std::all_of(q.begin(), q.end(), [](double value) { return std::isfinite(value); }) &&
                        std::all_of(velocity.begin(), velocity.end(), [](double value) { return std::isfinite(value); }) &&
                        std::all_of(acceleration.begin(), acceleration.end(), [](double value) { return std::isfinite(value); }) &&
                        std::all_of(jerk.begin(), jerk.end(), [](double value) { return std::isfinite(value); });
    if (!finite) throw std::runtime_error("non-finite native trajectory state: " + path.string());
    if (!result.time.empty() && t <= result.time.back()) throw std::runtime_error("non-increasing native trajectory time: " + path.string());
    result.time.push_back(t);
    result.q.push_back(q);
    result.velocity.push_back(velocity);
    result.acceleration.push_back(acceleration);
    result.jerk.push_back(jerk);
  }
  if (result.q.size() < 2) throw std::runtime_error("trajectory has fewer than two states: " + path.string());
  return result;
}

void validate_native_trajectory_contract(const NativeTrajectory& trajectory, double jerk_bound)
{
  if (!std::isfinite(jerk_bound) || jerk_bound < 0.0)
    throw std::runtime_error("jerk bound must be finite and nonnegative");
  for (std::size_t index = 0; index < trajectory.time.size(); ++index) {
    for (std::size_t joint = 0; joint < kDofs; ++joint) {
      if (std::abs(trajectory.jerk[index][joint]) > jerk_bound)
        throw std::runtime_error("exported jerk exceeds configured bound at row " + std::to_string(index) +
                                 " joint " + std::to_string(joint + 1));
    }
  }
  for (std::size_t index = 0; index + 1 < trajectory.time.size(); ++index) {
    const double dt = trajectory.time[index + 1] - trajectory.time[index];
    for (std::size_t joint = 0; joint < kDofs; ++joint) {
      const double q0 = trajectory.q[index][joint];
      const double v0 = trajectory.velocity[index][joint];
      const double a0 = trajectory.acceleration[index][joint];
      const double q_center = q0 + v0 * dt + 0.5 * a0 * dt * dt;
      const double q_radius = jerk_bound * dt * dt * dt / 6.0;
      const double v_center = v0 + a0 * dt;
      const double v_radius = jerk_bound * dt * dt / 2.0;
      const double a_center = a0;
      const double a_radius = jerk_bound * dt;
      const auto contains_outward = [](double value, double center, double radius) {
        const double lower = std::nextafter(center - radius, -std::numeric_limits<double>::infinity());
        const double upper = std::nextafter(center + radius, std::numeric_limits<double>::infinity());
        return value >= lower && value <= upper;
      };
      if (!contains_outward(trajectory.q[index + 1][joint], q_center, q_radius) ||
          !contains_outward(trajectory.velocity[index + 1][joint], v_center, v_radius) ||
          !contains_outward(trajectory.acceleration[index + 1][joint], a_center, a_radius))
        throw std::runtime_error("endpoint state is inconsistent with configured jerk bound at interval " +
                                 std::to_string(index) + " joint " + std::to_string(joint + 1));
    }
  }
}

bool acm_allows(const collision_detection::AllowedCollisionMatrix& acm,
                const std::string& first, const std::string& second)
{
  collision_detection::AllowedCollision::Type type = collision_detection::AllowedCollision::NEVER;
  return acm.getAllowedCollision(first, second, type) && type == collision_detection::AllowedCollision::ALWAYS;
}

double geometry_radius(const collision_detection::FCLGeometryConstPtr& geometry)
{
  const auto& aabb = geometry->collision_geometry_->aabb_local;
  double maximum = 0.0;
  for (int axis = 0; axis < 3; ++axis)
    maximum = std::max(maximum, std::max(std::abs(aabb.min_[axis]), std::abs(aabb.max_[axis])));
  return std::sqrt(3.0) * maximum;
}

std::vector<ShapeRef> make_shapes(const moveit::core::RobotModelConstPtr& model)
{
  std::vector<ShapeRef> result;
  for (const auto* link : model->getLinkModelsWithCollisionGeometry()) {
    if (!link) continue;
    const auto& origins = link->getCollisionOriginTransforms();
    for (std::size_t shape_index = 0; shape_index < link->getShapes().size(); ++shape_index) {
      auto geometry = collision_detection::createCollisionGeometry(link->getShapes()[shape_index], link, static_cast<int>(shape_index));
      if (!geometry || !geometry->collision_geometry_) throw std::runtime_error("FCL geometry construction failed: " + link->getName());
      const double origin_radius = shape_index < origins.size() ? origins[shape_index].translation().norm() : 0.0;
      const double radius = origin_radius + geometry_radius(geometry);
      result.push_back({link->getName(), shape_index, link, std::move(geometry), radius});
    }
  }
  if (result.empty()) throw std::runtime_error("robot has no collision geometry");
  return result;
}

std::vector<LinkInfo> make_links(const std::vector<ShapeRef>& shapes,
                                 const std::vector<const moveit::core::JointModel*>& active_joints)
{
  std::map<std::string, LinkInfo> by_name;
  for (std::size_t shape_index = 0; shape_index < shapes.size(); ++shape_index) {
    auto& link = by_name[shapes[shape_index].link];
    link.name = shapes[shape_index].link;
    link.model = shapes[shape_index].link_model;
    link.shapes.push_back(shape_index);
  }
  std::vector<LinkInfo> result;
  for (auto& entry : by_name) {
    auto link = entry.second;
    double tail_radius = 0.0;
    for (const auto shape_index : link.shapes) tail_radius = std::max(tail_radius, shapes[shape_index].local_radius);
    const auto* cursor = link.model;
    while (cursor && cursor->getParentLinkModel()) {
      const auto* joint = cursor->getParentJointModel();
      const double segment = cursor->getJointOriginTransform().translation().norm();
      if (joint) {
        for (std::size_t index = 0; index < active_joints.size(); ++index) {
          if (active_joints[index] != joint) continue;
          const double gain = joint->getType() == moveit::core::JointModel::PRISMATIC ? 1.0 : tail_radius + segment;
          link.motion_coefficient[index] = std::max(link.motion_coefficient[index], gain);
        }
      }
      tail_radius += segment;
      cursor = cursor->getParentLinkModel();
    }
    result.push_back(std::move(link));
  }
  return result;
}

std::vector<PairInfo> make_pairs(const std::vector<LinkInfo>& links,
                                 const std::vector<ShapeRef>& shapes,
                                 const collision_detection::AllowedCollisionMatrix& acm)
{
  std::vector<PairInfo> result;
  for (std::size_t first = 0; first < links.size(); ++first) {
    for (std::size_t second = first + 1; second < links.size(); ++second) {
      PairInfo pair;
      pair.key = links[first].name + "|" + links[second].name;
      pair.first_link = first;
      pair.second_link = second;
      pair.allowed = acm_allows(acm, links[first].name, links[second].name);
      for (const auto first_shape : links[first].shapes)
        for (const auto second_shape : links[second].shapes) pair.shape_pairs.emplace_back(first_shape, second_shape);
      for (std::size_t joint = 0; joint < kDofs; ++joint)
        pair.motion_coefficient[joint] = links[first].motion_coefficient[joint] + links[second].motion_coefficient[joint];
      result.push_back(std::move(pair));
    }
  }
  return result;
}

double endpoint_distance(const PairInfo& pair, const std::vector<ShapeRef>& shapes, const RobotState& state)
{
  fcl::DistanceRequestd request(true, true, 0.0, 0.0, 1.0e-9, fcl::GST_LIBCCD);
  double minimum = std::numeric_limits<double>::infinity();
  for (const auto [first_index, second_index] : pair.shape_pairs) {
    const auto& first = shapes[first_index];
    const auto& second = shapes[second_index];
    const auto first_transform = collision_detection::transform2fcl(state.getCollisionBodyTransform(first.link_model, first.shape_index));
    const auto second_transform = collision_detection::transform2fcl(state.getCollisionBodyTransform(second.link_model, second.shape_index));
    fcl::DistanceResultd result;
    const double returned = fcl::distance(first.geometry->collision_geometry_.get(), first_transform,
                                           second.geometry->collision_geometry_.get(), second_transform, request, result);
    const double value = std::min(returned, result.min_distance);
    if (std::isfinite(value)) minimum = std::min(minimum, value);
  }
  return minimum;
}

StateDistances query_state_at_q(const std::array<double, kDofs>& q,
                                const moveit::core::RobotModelConstPtr& model,
                                const moveit::core::JointModelGroup* group,
                                const std::vector<ShapeRef>& shapes,
                                const std::vector<PairInfo>& pairs,
                                StateDistanceCache& cache)
{
  std::string key;
  key.reserve(kDofs * sizeof(double));
  for (const double value : q) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(value));
    key.append(reinterpret_cast<const char*>(&bits), sizeof(bits));
  }
  const auto cached = cache.find(key);
  if (cached != cache.end()) return cached->second;
  RobotState state(model);
  state.setJointGroupPositions(group, std::vector<double>(q.begin(), q.end()));
  state.update();
  StateDistances result;
  result.pair_distance.assign(pairs.size(), std::numeric_limits<double>::quiet_NaN());
  for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
    if (pairs[pair_index].allowed) continue;
    result.pair_distance[pair_index] = endpoint_distance(pairs[pair_index], shapes, state);
    if (!std::isfinite(result.pair_distance[pair_index])) result.finite = false;
  }
  cache.emplace(std::move(key), result);
  return result;
}

StateDistances query_state(const NativeTrajectory& trajectory, std::size_t index,
                           const moveit::core::RobotModelConstPtr& model,
                           const moveit::core::JointModelGroup* group,
                           const std::vector<ShapeRef>& shapes,
                           const std::vector<PairInfo>& pairs,
                           StateDistanceCache& cache)
{
  return query_state_at_q(trajectory.q[index], model, group, shapes, pairs, cache);
}

std::array<double, kDofs> endpoint_motion(const NativeTrajectory& trajectory, std::size_t index,
                                           double dt, double jerk_bound)
{
  std::array<double, kDofs> result{};
  for (std::size_t joint = 0; joint < kDofs; ++joint)
    result[joint] = std::abs(trajectory.velocity[index][joint]) * dt +
                    0.5 * std::abs(trajectory.acceleration[index][joint]) * dt * dt +
                    jerk_bound * dt * dt * dt / 6.0;
  return result;
}

double dot(const std::array<double, kDofs>& first, const std::array<double, kDofs>& second)
{
  double result = 0.0;
  for (std::size_t index = 0; index < kDofs; ++index) result += first[index] * second[index];
  return result;
}

IntervalEvaluation evaluate_interval(const NativeTrajectory& trajectory, std::size_t left, std::size_t right,
                                     const StateDistances& start, const StateDistances& end,
                                     const PairInfo& pair, std::size_t pair_index,
                                     const Args& args,
                                     const moveit::core::RobotModelConstPtr& model,
                                     const moveit::core::JointModelGroup* group,
                                     const std::vector<ShapeRef>& shapes,
                                     const std::vector<PairInfo>& pairs,
                                     StateDistanceCache& cache)
{
  const double dt = trajectory.time[right] - trajectory.time[left];
  const auto start_motion = endpoint_motion(trajectory, left, dt, args.jerk_bound);
  const auto end_motion = endpoint_motion(trajectory, right, dt, args.jerk_bound);
  IntervalEvaluation result;
  result.start_distance = start.pair_distance[pair_index];
  result.end_distance = end.pair_distance[pair_index];
  result.motion_bound = std::max(dot(pair.motion_coefficient, start_motion), dot(pair.motion_coefficient, end_motion));
  if (!std::isfinite(result.start_distance) || !std::isfinite(result.end_distance) || !std::isfinite(result.motion_bound)) {
    result.status = Status::Unresolved;
    return result;
  }
  if (result.start_distance <= args.threshold + kNumericalTolerance || result.end_distance <= args.threshold + kNumericalTolerance) {
    result.status = Status::CollisionFound;
    result.lower_bound = std::min(result.start_distance, result.end_distance);
    return result;
  }

  // Add a conservative virtual midpoint witness.  The midpoint q is the
  // second-order Taylor state from the actual left endpoint.  For any true
  // C3 trajectory whose absolute joint jerk is bounded by J, the component
  // error from that witness over the whole interval is bounded by:
  //
  //   |v0| h + 3/2 |a0| h^2 + J (dt^3 + h^3) / 6,  h = dt / 2.
  //
  // This is a state-space enclosure, not an assumption that the virtual
  // midpoint is an exported/native state.  FK/FCL distance at the witness,
  // minus its pairwise link-motion Lipschitz radius, is therefore a valid
  // lower bound for the whole interval.  Taking the maximum with the two
  // endpoint-centered lower bounds materially tightens the certificate while
  // remaining fail-closed under the same jerk contract.
  const double half_dt = 0.5 * dt;
  std::array<double, kDofs> midpoint_q{};
  std::array<double, kDofs> midpoint_radius{};
  for (std::size_t joint = 0; joint < kDofs; ++joint) {
    const double velocity = trajectory.velocity[left][joint];
    const double acceleration = trajectory.acceleration[left][joint];
    midpoint_q[joint] = trajectory.q[left][joint] + velocity * half_dt +
                        0.5 * acceleration * half_dt * half_dt;
    midpoint_radius[joint] = std::abs(velocity) * half_dt +
                             1.5 * std::abs(acceleration) * half_dt * half_dt +
                             args.jerk_bound * (dt * dt * dt + half_dt * half_dt * half_dt) / 6.0;
  }
  const auto midpoint = query_state_at_q(midpoint_q, model, group, shapes, pairs, cache);
  const double midpoint_distance = midpoint.pair_distance[pair_index];
  const double midpoint_motion = dot(pair.motion_coefficient, midpoint_radius);
  if (!std::isfinite(midpoint_distance) || !std::isfinite(midpoint_motion)) {
    result.status = Status::Unresolved;
    return result;
  }
  const double start_lower_bound = result.start_distance - dot(pair.motion_coefficient, start_motion);
  const double end_lower_bound = result.end_distance - dot(pair.motion_coefficient, end_motion);
  const double midpoint_lower_bound = midpoint_distance - midpoint_motion;
  result.lower_bound = std::max({start_lower_bound, end_lower_bound, midpoint_lower_bound});
  result.status = result.lower_bound > args.threshold + kNumericalTolerance ? Status::CertifiedClear : Status::Unresolved;
  return result;
}

void assign_status(CaseResult& result, std::size_t pair_index, std::size_t left, std::size_t right, Status status)
{
  for (std::size_t interval = left; interval < right; ++interval) result.statuses[pair_index][interval] = status;
}

void refine_interval(CaseResult& result, const NativeTrajectory& trajectory, std::size_t left, std::size_t right,
                     std::size_t depth, const Args& args,
                     const std::vector<PairInfo>& pairs,
                     const std::vector<StateDistances>& states,
                     const moveit::core::RobotModelConstPtr& model,
                     const moveit::core::JointModelGroup* group,
                     const std::vector<ShapeRef>& shapes,
                     StateDistanceCache& cache)
{
  result.max_refinement_depth = std::max(result.max_refinement_depth, depth);
  bool needs_split = false;
  std::vector<IntervalEvaluation> evaluations;
  evaluations.reserve(pairs.size());
  for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
    if (pairs[pair_index].allowed) {
      evaluations.push_back({Status::CertifiedClear, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), 0.0});
      continue;
    }
    const auto evaluation = evaluate_interval(trajectory, left, right, states[left], states[right], pairs[pair_index], pair_index,
                                              args, model, group, shapes, pairs, cache);
    if (evaluation.status == Status::Unresolved && right - left > 1 && depth < args.max_depth) needs_split = true;
    if (evaluation.status == Status::CollisionFound) assign_status(result, pair_index, left, right, Status::CollisionFound);
    evaluations.push_back(evaluation);
  }
  if (needs_split) {
    const auto middle = left + (right - left) / 2;
    refine_interval(result, trajectory, left, middle, depth + 1, args, pairs, states, model, group, shapes, cache);
    refine_interval(result, trajectory, middle, right, depth + 1, args, pairs, states, model, group, shapes, cache);
    return;
  }
  for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
    if (pairs[pair_index].allowed) continue;
    if (evaluations[pair_index].status == Status::CertifiedClear)
      assign_status(result, pair_index, left, right, Status::CertifiedClear);
    else if (evaluations[pair_index].status == Status::Unresolved)
      assign_status(result, pair_index, left, right, Status::Unresolved);
    for (std::size_t interval = left; interval < right; ++interval) {
      result.lower_bounds[pair_index][interval] = evaluations[pair_index].lower_bound;
      result.start_distances[pair_index][interval] = evaluations[pair_index].start_distance;
      result.end_distances[pair_index][interval] = evaluations[pair_index].end_distance;
      result.motion_bounds[pair_index][interval] = evaluations[pair_index].motion_bound;
    }
  }
}

std::string rle_json(const std::vector<Status>& statuses, const std::vector<double>& times)
{
  std::ostringstream out;
  out << '[';
  std::size_t start = 0;
  bool first = true;
  while (start < statuses.size()) {
    std::size_t end = start + 1;
    while (end < statuses.size() && statuses[end] == statuses[start]) ++end;
    if (!first) out << ',';
    first = false;
    out << "{\"status\":" << json_string(status_string(statuses[start]))
        << ",\"interval_start\":" << start
        << ",\"interval_end_exclusive\":" << end
        << ",\"time_start_s\":" << number(times[start])
        << ",\"time_end_s\":" << number(times[end]) << '}';
    start = end;
  }
  out << ']';
  return out.str();
}

CaseResult run_case(const Case& item, const Args& args,
                    const moveit::core::RobotModelConstPtr& model,
                    const moveit::core::JointModelGroup* group,
                    const std::vector<ShapeRef>& shapes,
                    const std::vector<PairInfo>& pairs,
                    StateDistanceCache& cache)
{
  const NativeTrajectory trajectory = read_native_trajectory(item.trajectory);
  validate_native_trajectory_contract(trajectory, args.jerk_bound);
  CaseResult result;
  result.id = item.id;
  result.family = item.family;
  result.state_count = trajectory.q.size();
  result.interval_count = trajectory.q.size() - 1;
  result.statuses.assign(pairs.size(), std::vector<Status>(result.interval_count, Status::Unresolved));
  result.lower_bounds.assign(pairs.size(), std::vector<double>(result.interval_count, std::numeric_limits<double>::quiet_NaN()));
  result.start_distances.assign(pairs.size(), std::vector<double>(result.interval_count, std::numeric_limits<double>::quiet_NaN()));
  result.end_distances.assign(pairs.size(), std::vector<double>(result.interval_count, std::numeric_limits<double>::quiet_NaN()));
  result.motion_bounds.assign(pairs.size(), std::vector<double>(result.interval_count, std::numeric_limits<double>::quiet_NaN()));

  std::vector<StateDistances> states(trajectory.q.size());
  for (std::size_t index = 0; index < trajectory.q.size(); ++index) {
    states[index] = query_state(trajectory, index, model, group, shapes, pairs, cache);
    if ((index + 1) % 1000 == 0 || index + 1 == trajectory.q.size())
      std::cerr << "D54 FK-distance case=" << item.id << " states=" << (index + 1) << "/" << trajectory.q.size() << "\n";
  }
  for (std::size_t left = 0; left < result.interval_count; left += args.initial_stride) {
    const std::size_t right = std::min(left + args.initial_stride, result.interval_count);
    refine_interval(result, trajectory, left, right, 0, args, pairs, states, model, group, shapes, cache);
  }
  for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
    // ACM-allowed pairs are deliberately not measured.  Their status vectors
    // remain the initialization value and must never be counted as unresolved
    // required-pair regions in the aggregate certificate.
    if (pairs[pair_index].allowed) continue;
    for (std::size_t interval = 0; interval < result.interval_count; ++interval) {
      const auto status = result.statuses[pair_index][interval];
      if (status == Status::CertifiedClear) {
        ++result.certified_region_count;
        if (std::isfinite(result.lower_bounds[pair_index][interval]) && result.lower_bounds[pair_index][interval] < result.minimum_certified_clearance) {
          result.minimum_certified_clearance = result.lower_bounds[pair_index][interval];
          result.worst_pair = pairs[pair_index].key;
          result.worst_interval = interval;
          result.worst_t0 = trajectory.time[interval];
          result.worst_t1 = trajectory.time[interval + 1];
        }
      } else if (status == Status::CollisionFound) ++result.collision_region_count;
      else ++result.unresolved_region_count;
      result.minimum_observed_endpoint_distance = std::min(result.minimum_observed_endpoint_distance,
                                                            std::min(result.start_distances[pair_index][interval], result.end_distances[pair_index][interval]));
    }
  }
  return result;
}

void write_case(std::ofstream& output, const CaseResult& result, const NativeTrajectory& trajectory,
                const std::vector<PairInfo>& pairs, const Args& args)
{
  bool all_clear = result.collision_region_count == 0 && result.unresolved_region_count == 0;
  output << "{\"case_id\":" << json_string(result.id)
         << ",\"family\":" << json_string(result.family)
         << ",\"state_count\":" << result.state_count
         << ",\"interval_count\":" << result.interval_count
         << ",\"pair_count\":" << pairs.size()
         << ",\"required_pair_count\":" << std::count_if(pairs.begin(), pairs.end(), [](const PairInfo& pair) { return !pair.allowed; })
         << ",\"certified_region_count\":" << result.certified_region_count
         << ",\"collision_region_count\":" << result.collision_region_count
         << ",\"unresolved_region_count\":" << result.unresolved_region_count
         << ",\"max_refinement_depth\":" << result.max_refinement_depth
         << ",\"status\":" << json_string(all_clear ? "PASS" : result.collision_region_count ? "COLLISION_FOUND" : "UNRESOLVED")
         << ",\"strict_articulated_continuous_self_collision\":" << json_string(all_clear ? "PASS" : result.collision_region_count ? "COLLISION_FOUND" : "UNRESOLVED")
         << ",\"continuous_self_clearance_status\":" << json_string(all_clear ? "PASS" : "UNRESOLVED")
         << ",\"minimum_certified_clearance_m\":" << number(all_clear ? result.minimum_certified_clearance : std::numeric_limits<double>::quiet_NaN())
         << ",\"minimum_observed_endpoint_distance_m\":" << number(result.minimum_observed_endpoint_distance)
         << ",\"worst_certified_interval\":{\"pair\":" << json_string(result.worst_pair)
         << ",\"interval_index\":" << result.worst_interval
         << ",\"time_start_s\":" << number(result.worst_t0)
         << ",\"time_end_s\":" << number(result.worst_t1)
         << ",\"lower_bound_m\":" << number(result.minimum_certified_clearance) << "}"
         << ",\"pair_interval_runs\":[";
  bool first_pair = true;
  for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
    if (pairs[pair_index].allowed) continue;
    if (!first_pair) output << ',';
    first_pair = false;
    output << "{\"pair\":" << json_string(pairs[pair_index].key)
           << ",\"status_runs\":" << rle_json(result.statuses[pair_index], trajectory.time) << '}';
  }
  output << "]"
         << ",\"certificate_method\":\"fk_aware_adaptive_pairwise_interval_bound\""
         << ",\"motion_semantics\":\"actual_MoveIt2_RobotState_FK_at_exported_trajectory_endpoints_with_C3_jerk_bounded_joint_envelope\""
         << ",\"jerk_bound_rad_s3\":" << number(args.jerk_bound)
         << ",\"threshold_m\":" << number(args.threshold)
         << ",\"collision_method\":\"adaptive_discrete_interpolation\""
         << ",\"distance_backend\":\"FCL_0.7_signed_distance_at_FK_endpoint_states\""
         << ",\"status_semantics\":\"CERTIFIED_CLEAR is a conservative lower-bound proof under the recorded trajectory smoothness and jerk-bound assumptions; UNRESOLVED is never promoted to PASS\""
         << "}\n";
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    const Args args = parse_args(argc, argv);
    fs::create_directories(args.output);
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
    std::vector<const moveit::core::JointModel*> active_joints;
    for (const auto& name : group->getActiveJointModelNames()) active_joints.push_back(model->getJointModel(name));
    const auto shapes = make_shapes(model);
    auto links = make_links(shapes, active_joints);
    std::sort(links.begin(), links.end(), [](const LinkInfo& first, const LinkInfo& second) { return first.name < second.name; });
    auto scene = std::make_unique<planning_scene::PlanningScene>(model);
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
    const auto pairs = make_pairs(links, shapes, scene->getAllowedCollisionMatrix());
    const std::size_t required_pairs = std::count_if(pairs.begin(), pairs.end(), [](const PairInfo& pair) { return !pair.allowed; });
    std::ofstream summaries(fs::path(args.output) / "D54_CONTINUOUS_SELF_COLLISION_CASES.jsonl");
    if (!summaries) throw std::runtime_error("cannot create D54 case output");
    StateDistanceCache state_cache;
    std::size_t selected = 0;
    std::size_t passing = 0;
    std::size_t collisions = 0;
    std::size_t unresolved = 0;
    double global_min = std::numeric_limits<double>::infinity();
    std::string global_case;
    std::string global_pair;
    for (const auto& item : cases) {
      if (!args.case_filter.empty() && item.id != args.case_filter) continue;
      ++selected;
      std::cerr << "D54 articulated certificate case=" << item.id << "\n";
      const auto trajectory = read_native_trajectory(item.trajectory);
      validate_native_trajectory_contract(trajectory, args.jerk_bound);
      const auto result = run_case(item, args, model, group, shapes, pairs, state_cache);
      write_case(summaries, result, trajectory, pairs, args);
      if (result.collision_region_count == 0 && result.unresolved_region_count == 0) ++passing;
      collisions += result.collision_region_count;
      unresolved += result.unresolved_region_count;
      if (std::isfinite(result.minimum_certified_clearance) && result.minimum_certified_clearance < global_min) {
        global_min = result.minimum_certified_clearance;
        global_case = result.id;
        global_pair = result.worst_pair;
      }
    }
    if (selected == 0) throw std::runtime_error("case filter selected no manifest case");
    std::ofstream report(fs::path(args.output) / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json");
    report << "{\n"
           << "  \"schema_version\":\"d54-fk-aware-articulated-certificate-v1\",\n"
           << "  \"status\":\"" << (passing == selected && required_pairs > 0 ? "PASS" : collisions ? "COLLISION_FOUND" : "UNRESOLVED") << "\",\n"
           << "  \"case_count\":" << selected << ",\n"
           << "  \"passing_case_count\":" << passing << ",\n"
           << "  \"collision_region_count\":" << collisions << ",\n"
           << "  \"unresolved_region_count\":" << unresolved << ",\n"
           << "  \"collision_link_count\":" << links.size() << ",\n"
           << "  \"pair_universe_count\":" << pairs.size() << ",\n"
           << "  \"required_pair_count\":" << required_pairs << ",\n"
           << "  \"required_pair_coverage_complete\":" << (required_pairs > 0 ? "true" : "false") << ",\n"
           << "  \"all_required_pairs\":[";
    bool first = true;
    for (const auto& pair : pairs) {
      if (pair.allowed) continue;
      if (!first) report << ',';
      first = false;
      report << json_string(pair.key);
    }
    report << "],\n  \"allowed_pairs\":[";
    first = true;
    for (const auto& pair : pairs) {
      if (!pair.allowed) continue;
      if (!first) report << ',';
      first = false;
      report << json_string(pair.key);
    }
    report << "],\n"
           << "  \"minimum_certified_clearance_m\":" << number(passing == selected ? global_min : std::numeric_limits<double>::quiet_NaN()) << ",\n"
           << "  \"worst_case_id\":" << json_string(global_case) << ",\n"
           << "  \"worst_pair\":" << json_string(global_pair) << ",\n"
           << "  \"certificate_method\":\"fk_aware_adaptive_pairwise_interval_bound\",\n"
           << "  \"motion_semantics\":\"actual_articulated_FK(q(t))_bounded_by_exported_native_state_and_jerk_envelope\",\n"
           << "  \"assumptions\":{\n"
           << "    \"trajectory_state_source\":\"validated native post-Ruckig q,dq,ddq,jerk CSV from the supplied manifest\",\n"
           << "    \"input_contract_validation\":\"required 6D columns, finite values, strictly increasing time, exported jerk bound, and jerk-bounded endpoint reachability\",\n"
           << "    \"within_interval_smoothness\":\"C3 trajectory with per-joint absolute jerk bounded by configured native Ruckig limit\",\n"
           << "    \"joint_motion_bound\":\"Taylor remainder bound from actual interval endpoint FK states\",\n"
           << "    \"link_motion_bound\":\"configuration-independent joint-axis radius bound from complete collision geometry\",\n"
           << "    \"narrow_phase\":\"FCL signed distance for every shape pair at actual FK endpoint states\",\n"
           << "    \"numerical_acceptance_margin_m\":" << number(kNumericalTolerance) << ",\n"
           << "    \"threshold_m\":" << number(args.threshold) << ",\n"
           << "    \"jerk_bound_rad_s3\":" << number(args.jerk_bound) << "\n"
           << "  },\n"
           << "  \"collision_method\":\"adaptive_discrete_interpolation\",\n"
           << "  \"continuous_self_clearance_semantics\":\"certified conservative model-based lower bound; hardware clearance not available\",\n"
           << "  \"unresolved_is_not_pass\":true\n"
           << "}\n";
    report.close();
    summaries.close();
    rclcpp::shutdown();
    return passing == selected && required_pairs > 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "stage4f_articulated_certificate ERROR: " << error.what() << "\n";
    if (rclcpp::ok()) rclcpp::shutdown();
    return 1;
  }
}
