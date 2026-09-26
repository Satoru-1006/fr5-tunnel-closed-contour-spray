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
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <fcl/narrowphase/continuous_collision.h>
#include <fcl/math/motion/screw_motion.h>
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection/collision_matrix.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <moveit/collision_detection_fcl/collision_env_fcl.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <srdfdom/srdfdom/model.h>
#include <urdf/model.h>

namespace fs = std::filesystem;
using moveit::core::RobotState;

namespace {

constexpr double kTimeTolerance = 1.0e-12;

struct Args {
  std::string manifest;
  std::string urdf;
  std::string srdf;
  std::string output;
  std::string group{"fairino5_v6_group"};
  std::size_t max_segments{0};
  std::string case_filter;
};

struct Case {
  std::string id;
  std::string trajectory;
  std::string family;
};

struct NativeTrajectory {
  std::vector<double> time;
  std::vector<std::array<double, 6>> q;
};

struct ShapeRef {
  std::string link;
  std::size_t shape_index{0};
  const moveit::core::LinkModel* link_model{nullptr};
  collision_detection::FCLGeometryConstPtr geometry;
};

struct IntervalShape {
  fcl::Transform3d begin_transform;
  fcl::Transform3d end_transform;
  fcl::AABBd swept_bound;
};

struct PairQuery {
  bool broadphase_rejected{false};
  double continuous_return{0.0};
  fcl::ContinuousCollisionResultd result;
};

struct RunStats {
  std::size_t interval_count{0};
  std::size_t swept_pair_calls{0};
  std::size_t broadphase_rejected_pairs{0};
  std::size_t continuous_collision_count{0};
  std::size_t continuous_api_error_count{0};
  std::size_t endpoint_contact_count{0};
  std::set<std::string> checked_pairs;
  std::set<std::string> allowed_pairs;
  std::set<std::string> unsupported_pairs;
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

std::string json_array(const std::array<double, 6>& values)
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

std::string json_array(const std::vector<std::string>& values)
{
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << json_string(values[i]);
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
    result.push_back(item);
  }
  if (!line.empty() && line.back() == ',') result.emplace_back();
  return result;
}

Args parse_args(int argc, char** argv)
{
  Args args;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key(argv[i]);
    const std::string value(argv[i + 1]);
    if (key == "--manifest") args.manifest = value;
    else if (key == "--urdf") args.urdf = value;
    else if (key == "--srdf") args.srdf = value;
    else if (key == "--output") args.output = value;
    else if (key == "--group") args.group = value;
    else if (key == "--max-segments") args.max_segments = static_cast<std::size_t>(std::stoull(value));
    else if (key == "--case") args.case_filter = value;
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (args.manifest.empty() || args.urdf.empty() || args.srdf.empty() || args.output.empty())
    throw std::runtime_error("missing --manifest/--urdf/--srdf/--output");
  return args;
}

std::map<std::string, std::size_t> header_map(const std::vector<std::string>& header)
{
  std::map<std::string, std::size_t> columns;
  for (std::size_t i = 0; i < header.size(); ++i) columns[header[i]] = i;
  return columns;
}

std::size_t required_column(const std::map<std::string, std::size_t>& columns, const std::string& name)
{
  const auto it = columns.find(name);
  if (it == columns.end()) throw std::runtime_error("missing CSV column: " + name);
  return it->second;
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
  std::array<std::size_t, 6> q_columns{};
  for (int joint = 0; joint < 6; ++joint) q_columns[joint] = required_column(columns, "j" + std::to_string(joint + 1) + "_q");
  NativeTrajectory result;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split_csv(line);
    if (fields.size() <= *std::max_element(q_columns.begin(), q_columns.end())) throw std::runtime_error("short native trajectory row");
    const double t = std::stod(fields[time]);
    std::array<double, 6> q{};
    for (int joint = 0; joint < 6; ++joint) q[joint] = std::stod(fields[q_columns[joint]]);
    if (!std::isfinite(t) || !std::all_of(q.begin(), q.end(), [](double x) { return std::isfinite(x); }))
      throw std::runtime_error("non-finite native trajectory state: " + path.string());
    if (!result.time.empty() && t <= result.time.back()) throw std::runtime_error("non-increasing native trajectory time: " + path.string());
    result.time.push_back(t);
    result.q.push_back(q);
  }
  if (result.q.size() < 2) throw std::runtime_error("trajectory has fewer than two states: " + path.string());
  return result;
}

bool acm_allows(const collision_detection::AllowedCollisionMatrix& acm, const std::string& first,
                const std::string& second)
{
  collision_detection::AllowedCollision::Type type = collision_detection::AllowedCollision::NEVER;
  if (acm.getAllowedCollision(first, second, type) && type == collision_detection::AllowedCollision::ALWAYS) return true;
  // Conditional entries cannot be evaluated without a MoveIt contact predicate.
  // They are therefore checked conservatively instead of being silently skipped.
  return false;
}

bool aabb_disjoint(const fcl::AABBd& first, const fcl::AABBd& second)
{
  for (int axis = 0; axis < 3; ++axis) {
    if (first.max_[axis] < second.min_[axis] || second.max_[axis] < first.min_[axis]) return true;
  }
  return false;
}

fcl::AABBd swept_aabb(const collision_detection::FCLGeometryConstPtr& moveit_geometry,
                     const fcl::Transform3d& begin,
                     const fcl::Transform3d& end)
{
  // Reuse FCL's own Taylor-model motion bound for the same ScrewMotion that
  // the exact continuous query uses.  This avoids an ad-hoc sampled bound and
  // preserves a rigorous broadphase rejection for the entire interval.
  auto geometry = std::const_pointer_cast<fcl::CollisionGeometryd>(moveit_geometry->collision_geometry_);
  auto motion = std::make_shared<fcl::ScrewMotion<double>>(begin, end);
  fcl::ContinuousCollisionObjectd object(geometry, motion);
  object.computeAABB();
  return object.getAABB();
}

std::vector<ShapeRef> make_shapes(const moveit::core::RobotModelConstPtr& model,
                                  const moveit::core::JointModelGroup* group)
{
  const std::unordered_set<std::string> group_links(group->getLinkModelNames().begin(), group->getLinkModelNames().end());
  std::vector<ShapeRef> result;
  for (const auto* link : model->getLinkModelsWithCollisionGeometry()) {
    if (!link || group_links.find(link->getName()) == group_links.end()) continue;
    for (std::size_t shape_index = 0; shape_index < link->getShapes().size(); ++shape_index) {
      auto geometry = collision_detection::createCollisionGeometry(link->getShapes()[shape_index], link, static_cast<int>(shape_index));
      if (!geometry || !geometry->collision_geometry_) throw std::runtime_error("MoveIt FCL geometry construction failed for " + link->getName());
      result.push_back({link->getName(), shape_index, link, std::move(geometry)});
    }
  }
  if (result.empty()) throw std::runtime_error("no group collision geometry was constructed");
  return result;
}

std::string link_pair(const ShapeRef& first, const ShapeRef& second)
{
  return first.link < second.link ? first.link + "|" + second.link : second.link + "|" + first.link;
}

void write_backend_provenance(const fs::path& path, const Args& args,
                              const moveit::core::RobotModelConstPtr& model,
                              const moveit::core::JointModelGroup* group,
                              const std::vector<ShapeRef>& shapes,
                              const collision_detection::AllowedCollisionMatrix& acm)
{
  std::ofstream output(path);
  output << "{\n"
         << "  \"schema_version\":\"d49-c2-fcl-continuous-self-collision-v1\",\n"
         << "  \"status\":\"NATIVE_BACKEND_READY\",\n"
         << "  \"backend\":\"FCL_0.7_continuousCollide\",\n"
         << "  \"ccd_motion_type\":\"CCDM_SCREW\",\n"
         << "  \"ccd_solver_type\":\"CCDC_CONSERVATIVE_ADVANCEMENT\",\n"
         << "  \"gjk_solver_type\":\"GST_LIBCCD\",\n"
         << "  \"toc_err\":1e-9,\n"
         << "  \"num_max_iterations\":100,\n"
         << "  \"robot_model_name\":" << json_string(model->getName()) << ",\n"
         << "  \"model_frame\":" << json_string(model->getModelFrame()) << ",\n"
         << "  \"planning_group\":" << json_string(args.group) << ",\n"
         << "  \"active_joint_names\":" << json_array(group->getActiveJointModelNames()) << ",\n"
         << "  \"group_link_names\":" << json_array(group->getLinkModelNames()) << ",\n"
         << "  \"acm_source\":\"PlanningScene::getAllowedCollisionMatrix from supplied SRDF\",\n"
         << "  \"acm_entry_count\":" << acm.getSize() << ",\n"
         << "  \"continuous_distance_information\":\"not_available\",\n"
         << "  \"penetration_information\":\"not_available\",\n"
         << "  \"shape_count\":" << shapes.size() << "\n"
         << "}\n";
  std::ofstream shape_output(path.parent_path() / "continuous_self_collision_geometry.jsonl");
  for (const auto& shape : shapes) {
    shape_output << "{\"link\":" << json_string(shape.link)
                 << ",\"shape_index\":" << shape.shape_index
                 << ",\"geometry_node_type\":" << static_cast<int>(shape.geometry->collision_geometry_->getNodeType())
                 << "}\n";
  }
  std::ofstream acm_output(path.parent_path() / "continuous_self_collision_acm_pairs.jsonl");
  std::set<std::string> emitted;
  for (const auto& first : shapes) for (const auto& second : shapes) {
    if (first.link >= second.link) continue;
    const auto pair = link_pair(first, second);
    if (!emitted.insert(pair).second) continue;
    acm_output << "{\"pair\":" << json_string(pair)
               << ",\"allowed\":" << (acm_allows(acm, first.link, second.link) ? "true" : "false")
               << "}\n";
  }
}

void write_contact(std::ofstream& output, const Case& item, std::size_t segment,
                   const NativeTrajectory& trajectory, const ShapeRef& first, const ShapeRef& second,
                   const fcl::ContinuousCollisionResultd& result, double continuous_return)
{
  output << "{\"case_id\":" << json_string(item.id)
         << ",\"family\":" << json_string(item.family)
         << ",\"segment_index\":" << segment
         << ",\"time_start_s\":" << number(trajectory.time[segment])
         << ",\"time_end_s\":" << number(trajectory.time[segment + 1])
         << ",\"link_pair\":" << json_string(link_pair(first, second))
         << ",\"link_1\":" << json_string(first.link)
         << ",\"link_1_shape_index\":" << first.shape_index
         << ",\"link_2\":" << json_string(second.link)
         << ",\"link_2_shape_index\":" << second.shape_index
         << ",\"q_start\":" << json_array(trajectory.q[segment])
         << ",\"q_end\":" << json_array(trajectory.q[segment + 1])
         << ",\"collision\":" << (result.is_collide ? "true" : "false")
         << ",\"contact_fraction\":" << number(result.is_collide ? result.time_of_contact : std::numeric_limits<double>::quiet_NaN())
         << ",\"contact_time_s\":" << number(result.is_collide ? trajectory.time[segment] + result.time_of_contact * (trajectory.time[segment + 1] - trajectory.time[segment]) : std::numeric_limits<double>::quiet_NaN())
         << ",\"penetration_depth_m\":null"
         << ",\"distance_m\":null"
         << ",\"backend\":\"FCL_0.7_continuousCollide\""
         << ",\"continuous_return\":" << number(continuous_return)
         << ",\"api_status\":\"" << (continuous_return >= 0.0 ? "OK" : "ERROR") << "\"}\n";
}

int run_case(const Case& item, const Args& args, const moveit::core::RobotModelConstPtr& model,
             const moveit::core::JointModelGroup* group, const std::vector<ShapeRef>& shapes,
             const collision_detection::AllowedCollisionMatrix& acm, std::ofstream& contacts,
             std::ofstream& summaries, RunStats& total)
{
  const NativeTrajectory trajectory = read_native_trajectory(item.trajectory);
  RobotState begin(model), end(model);
  std::size_t intervals = trajectory.q.size() - 1;
  if (args.max_segments > 0) intervals = std::min(intervals, args.max_segments);
  std::size_t calls = 0;
  std::size_t broadphase_rejected = 0;
  std::size_t collisions = 0;
  std::size_t api_errors = 0;
  std::size_t endpoint_contacts = 0;
  std::set<std::string> case_pairs;
  fcl::ContinuousCollisionRequestd request(100, 1.0e-9, fcl::CCDM_SCREW, fcl::GST_LIBCCD, fcl::CCDC_CONSERVATIVE_ADVANCEMENT);

  std::vector<std::pair<std::size_t, std::size_t>> shape_pairs;
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    for (std::size_t j = i + 1; j < shapes.size(); ++j) {
      if (shapes[i].link == shapes[j].link) continue;
      if (acm_allows(acm, shapes[i].link, shapes[j].link)) {
        total.allowed_pairs.insert(link_pair(shapes[i], shapes[j]));
        continue;
      }
      shape_pairs.emplace_back(i, j);
    }
  }

  for (std::size_t segment = 0; segment < intervals; ++segment) {
    begin.setJointGroupPositions(group, std::vector<double>(trajectory.q[segment].begin(), trajectory.q[segment].end()));
    end.setJointGroupPositions(group, std::vector<double>(trajectory.q[segment + 1].begin(), trajectory.q[segment + 1].end()));
    begin.updateCollisionBodyTransforms();
    end.updateCollisionBodyTransforms();
    ++total.interval_count;
    std::vector<IntervalShape> interval_shapes;
    interval_shapes.reserve(shapes.size());
    for (const auto& shape : shapes) {
      const auto begin_transform = collision_detection::transform2fcl(
          begin.getCollisionBodyTransform(shape.link_model, shape.shape_index));
      const auto end_transform = collision_detection::transform2fcl(
          end.getCollisionBodyTransform(shape.link_model, shape.shape_index));
      interval_shapes.push_back({begin_transform, end_transform,
                                 swept_aabb(shape.geometry, begin_transform, end_transform)});
    }
    std::vector<PairQuery> pair_results(shape_pairs.size());
    // Keep exact FCL traversal serial within one process.  An isolated
    // OpenMP experiment showed that this FCL build is not re-entrant during
    // continuous BVH traversal; the campaign launcher may use separate
    // processes, but one verifier process remains serial and deterministic.
    for (std::size_t pair_index = 0; pair_index < shape_pairs.size(); ++pair_index) {
      const auto& shape_pair = shape_pairs[pair_index];
      const auto& first_motion = interval_shapes[shape_pair.first];
      const auto& second_motion = interval_shapes[shape_pair.second];
      auto& pair_result = pair_results[pair_index];
      if (aabb_disjoint(first_motion.swept_bound, second_motion.swept_bound)) {
        pair_result.broadphase_rejected = true;
        continue;
      }
      const auto& first = shapes[shape_pair.first];
      const auto& second = shapes[shape_pair.second];
      pair_result.continuous_return = fcl::continuousCollide(
          first.geometry->collision_geometry_.get(), first_motion.begin_transform, first_motion.end_transform,
          second.geometry->collision_geometry_.get(), second_motion.begin_transform, second_motion.end_transform,
          request, pair_result.result);
    }
    for (std::size_t pair_index = 0; pair_index < shape_pairs.size(); ++pair_index) {
      const auto& shape_pair = shape_pairs[pair_index];
      const auto& first = shapes[shape_pair.first];
      const auto& second = shapes[shape_pair.second];
      const auto pair = link_pair(first, second);
      case_pairs.insert(pair);
      total.checked_pairs.insert(pair);
      const auto& pair_result = pair_results[pair_index];
      if (pair_result.broadphase_rejected) {
        ++broadphase_rejected;
        ++total.broadphase_rejected_pairs;
        continue;
      }
      ++calls;
      ++total.swept_pair_calls;
      if (pair_result.continuous_return < 0.0) {
        ++api_errors;
        ++total.continuous_api_error_count;
        total.unsupported_pairs.insert(pair);
      }
      if (pair_result.result.is_collide) {
        ++collisions;
        ++total.continuous_collision_count;
        if (pair_result.result.time_of_contact <= kTimeTolerance || pair_result.result.time_of_contact >= 1.0 - kTimeTolerance) ++endpoint_contacts;
        ++total.endpoint_contact_count;
        write_contact(contacts, item, segment, trajectory, first, second, pair_result.result, pair_result.continuous_return);
      }
    }
    if ((segment + 1) % 1000 == 0 || segment + 1 == intervals) {
      std::cerr << "C2 FCL progress case=" << item.id << " intervals=" << (segment + 1)
                << "/" << intervals << " exact_calls=" << calls << " collisions=" << collisions << "\n";
    }
  }
  const bool complete = intervals == trajectory.q.size() - 1;
  const bool pass = api_errors == 0 && collisions == 0;
  std::vector<std::string> pair_list(case_pairs.begin(), case_pairs.end());
  summaries << "{\"case_id\":" << json_string(item.id)
            << ",\"family\":" << json_string(item.family)
            << ",\"state_count\":" << trajectory.q.size()
            << ",\"swept_interval_count\":" << intervals
            << ",\"complete_trajectory\":" << (complete ? "true" : "false")
            << ",\"trajectory_duration_s\":" << number(trajectory.time.back())
            << ",\"swept_pair_calls\":" << calls
            << ",\"broadphase_rejected_pairs\":" << broadphase_rejected
            << ",\"continuous_collision_count\":" << collisions
            << ",\"continuous_api_error_count\":" << api_errors
            << ",\"endpoint_contact_count\":" << endpoint_contacts
            << ",\"checked_link_pairs\":" << json_array(pair_list)
            << ",\"collision_method\":\"FCL_continuous_swept_link_geometry\""
            << ",\"continuous_self_collision_status\":"
            << json_string(pass ? (complete ? "CERTIFIED_ZERO_CONTACTS" : "SMOKE_ZERO_CONTACTS_NOT_COMPLETE") : "NOT_CERTIFIED")
            << ",\"status\":" << json_string(pass ? (complete ? "PASS" : "PASS_PARTIAL_SMOKE") : (api_errors > 0 ? "BLOCKED_BACKEND_ERROR" : "FAIL_CONTACT"))
            << "}\n";
  return pass ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    const Args args = parse_args(argc, argv);
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
    auto scene = std::make_unique<planning_scene::PlanningScene>(model);
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
    const auto& acm = scene->getAllowedCollisionMatrix();
    const auto shapes = make_shapes(model, group);
    const fs::path output(args.output);
    fs::create_directories(output);
    write_backend_provenance(output / "continuous_self_collision_backend.json", args, model, group, shapes, acm);
    std::ofstream contacts(output / "continuous_self_collision_contacts.jsonl");
    std::ofstream summaries(output / "continuous_self_collision_case_summary.jsonl");
    if (!contacts || !summaries) throw std::runtime_error("cannot create C2 evidence files");
    const auto cases = read_manifest(args.manifest);
    RunStats total;
    std::size_t selected = 0;
    std::size_t failed = 0;
    for (const auto& item : cases) {
      if (!args.case_filter.empty() && args.case_filter != item.id) continue;
      ++selected;
      std::cerr << "C2 FCL continuous case " << item.id << "\n";
      if (run_case(item, args, model, group, shapes, acm, contacts, summaries, total) != 0) ++failed;
    }
    if (selected == 0) throw std::runtime_error("case filter selected no manifest case");
    std::ofstream summary(output / "continuous_self_collision_summary.json");
    summary << "{\n"
            << "  \"schema_version\":\"d49-c2-fcl-continuous-self-collision-v1\",\n"
            << "  \"status\":" << json_string(failed == 0 ? "PASS" : "BLOCKED_OR_CONTACT") << ",\n"
            << "  \"measurement_backend\":\"FCL_0.7_continuousCollide\",\n"
            << "  \"continuous_self_collision_status\":" << json_string(failed == 0 ? "PASS_ZERO_CONTACTS" : "NOT_CERTIFIED") << ",\n"
            << "  \"case_count\":" << selected << ",\n"
            << "  \"swept_interval_count\":" << total.interval_count << ",\n"
            << "  \"swept_pair_call_count\":" << total.swept_pair_calls << ",\n"
            << "  \"broadphase_rejected_pair_count\":" << total.broadphase_rejected_pairs << ",\n"
            << "  \"continuous_collision_count\":" << total.continuous_collision_count << ",\n"
            << "  \"continuous_api_error_count\":" << total.continuous_api_error_count << ",\n"
            << "  \"endpoint_contact_count\":" << total.endpoint_contact_count << ",\n"
            << "  \"checked_link_pairs\":" << json_array(std::vector<std::string>(total.checked_pairs.begin(), total.checked_pairs.end())) << ",\n"
            << "  \"acm_allowed_link_pairs\":" << json_array(std::vector<std::string>(total.allowed_pairs.begin(), total.allowed_pairs.end())) << ",\n"
            << "  \"unsupported_link_pairs\":" << json_array(std::vector<std::string>(total.unsupported_pairs.begin(), total.unsupported_pairs.end())) << ",\n"
            << "  \"contact_distance\":\"not_available\",\n"
            << "  \"penetration_depth\":\"not_available\",\n"
            << "  \"scope\":\"Stage 0/1 ON-state open-arch only; persisted D48 post-Ruckig trajectories\"\n"
            << "}\n";
    contacts.close();
    summaries.close();
    scene.reset();
    rclcpp::shutdown();
    return failed == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "stage4d_continuous_self_collision ERROR: " << error.what() << "\n";
    if (rclcpp::ok()) rclcpp::shutdown();
    return 1;
  }
}
