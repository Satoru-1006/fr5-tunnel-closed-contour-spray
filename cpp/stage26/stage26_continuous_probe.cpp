// Stage 2.6 native continuous robot-world probe.
//
// The helper include is the same native MoveIt scene/mesh construction used by
// the frozen Stage 2.3B/2.4 evidence. The formal call made below is the
// two-state CollisionEnv::checkRobotCollision overload; no interpolation is
// substituted for that call.
#define main stage26_stage23a7_original_main
#include "../stage23a7/stage23a7_runtime_audit_probe.cpp"
#undef main

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include <moveit/collision_detection/collision_matrix.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

struct Interval {
  int index = -1;
  double t0 = 0.0;
  double t1 = 0.0;
  std::string process_order;
  std::string spray_state;
  std::string process_kind;
  std::string segment_id;
  std::string transition_id;
  std::string source_boundary;
  std::vector<double> q0;
  std::vector<double> q1;
};

static std::vector<Interval> read_intervals(const fs::path& path)
{
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("interval CSV open failed: " + path.string());
  std::string line;
  if (!std::getline(in, line))
    throw std::runtime_error("empty interval CSV");
  const auto header = split_csv(line);
  std::map<std::string, size_t> col;
  for (size_t i = 0; i < header.size(); ++i) {
    auto name = header[i];
    if (!name.empty() && name.back() == '\r')
      name.pop_back();
    col[name] = i;
  }
  const auto required = [&](const std::string& name) -> size_t {
    auto it = col.find(name);
    if (it == col.end())
      throw std::runtime_error("missing interval column: " + name);
    return it->second;
  };
  const size_t index = required("interval_index");
  const size_t t0 = required("time_start_s");
  const size_t t1 = required("time_end_s");
  const size_t order = required("process_order_index");
  const size_t state = required("spray_state");
  const size_t kind = required("process_kind");
  const size_t segment = required("segment_id");
  const size_t transition = required("transition_id");
  const size_t boundary = required("source_boundary");
  std::array<size_t, 6> q0{};
  std::array<size_t, 6> q1{};
  for (int j = 0; j < 6; ++j) {
    q0[j] = required("q0_" + std::to_string(j));
    q1[j] = required("q1_" + std::to_string(j));
  }
  std::vector<Interval> rows;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    const auto row = split_csv(line);
    Interval item;
    item.index = static_cast<int>(num(row[index]));
    item.t0 = num(row[t0]);
    item.t1 = num(row[t1]);
    item.process_order = row[order];
    item.spray_state = row[state];
    item.process_kind = row[kind];
    item.segment_id = row[segment];
    item.transition_id = row[transition];
    item.source_boundary = row[boundary];
    item.q0.resize(6);
    item.q1.resize(6);
    for (int j = 0; j < 6; ++j) {
      item.q0[j] = num(row[q0[j]]);
      item.q1[j] = num(row[q1[j]]);
    }
    rows.push_back(std::move(item));
  }
  return rows;
}

static void write_double_array(std::ostream& out, const std::vector<double>& values)
{
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i)
      out << ',';
    out << jn(values[i]);
  }
  out << ']';
}

static void write_string_or_null(std::ostream& out, const std::string& value)
{
  if (value.empty())
    out << "null";
  else
    js(out, value);
}

static moveit_msgs::msg::CollisionObject box_object(const std::string& id, const Eigen::Vector3d& center,
                                                    double side)
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

static collision_detection::CollisionResult run_world_collision(const collision_detection::CollisionEnv* env,
                                                                const collision_detection::CollisionRequest& request,
                                                                const RobotState& state,
                                                                const collision_detection::AllowedCollisionMatrix& acm)
{
  collision_detection::CollisionResult result;
  result.clear();
  env->checkRobotCollision(request, result, state, acm);
  return result;
}

static collision_detection::CollisionResult run_world_ccd(const collision_detection::CollisionEnv* env,
                                                          const collision_detection::CollisionRequest& request,
                                                          const RobotState& state0, const RobotState& state1,
                                                          const collision_detection::AllowedCollisionMatrix& acm)
{
  collision_detection::CollisionResult result;
  result.clear();
  env->checkRobotCollision(request, result, state0, state1, acm);
  return result;
}

static collision_detection::CollisionResult run_self_collision(const planning_scene::PlanningScene& scene,
                                                               const collision_detection::CollisionRequest& request,
                                                               const RobotState& state,
                                                               const collision_detection::AllowedCollisionMatrix& acm)
{
  collision_detection::CollisionResult result;
  result.clear();
  scene.checkSelfCollision(request, result, state, acm);
  return result;
}

// Stage 2.6 only needs the first native CCD contact to decide the interval.
// Keep contact collection bounded while preserving the Bullet two-state verdict.
static collision_detection::CollisionRequest stage26_request(const std::string& group)
{
  auto r = req(group, false);
  r.contacts = true;
  r.max_contacts = 1;
  r.max_contacts_per_pair = 1;
  r.pad_environment_collisions = true;
  r.pad_self_collisions = false;
  r.distance = false;
  return r;
}

static std::unique_ptr<planning_scene::PlanningScene> synthetic_scene(
    const moveit::core::RobotModelConstPtr& model, const Eigen::Vector3d& center, double side,
    const std::string& backend)
{
  auto scene = std::make_unique<planning_scene::PlanningScene>(model);
  scene->processCollisionObjectMsg(box_object("stage26_synthetic_obstacle", center, side));
  if (backend == "bullet")
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create());
  else
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  return scene;
}

static std::vector<std::pair<std::vector<double>, std::vector<double>>> synthetic_joint_pairs()
{
  std::vector<std::pair<std::vector<double>, std::vector<double>>> pairs_out;
  for (int joint = 0; joint < 6; ++joint) {
    std::vector<double> q0(6, 0.0), q1(6, 0.0);
    q0[joint] = -1.25;
    q1[joint] = 1.25;
    pairs_out.emplace_back(q0, q1);
  }
  for (int joint = 0; joint < 6; ++joint) {
    std::vector<double> q0(6, 0.0), q1(6, 0.0);
    q0[joint] = 1.25;
    q1[joint] = -1.25;
    pairs_out.emplace_back(q0, q1);
  }
  std::vector<double> q0(6, 0.0), q1(6, 0.0);
  q0[0] = -1.0;
  q0[1] = -0.7;
  q0[2] = 0.8;
  q1[0] = 1.0;
  q1[1] = 0.7;
  q1[2] = -0.8;
  pairs_out.emplace_back(q0, q1);
  return pairs_out;
}

static bool find_swept_positive_control(const moveit::core::RobotModelConstPtr& model, const std::string& group,
                                        std::ofstream& out, std::vector<double>& found_q0,
                                        std::vector<double>& found_q1, std::string& found_link,
                                        Eigen::Vector3d& found_center, double& found_side)
{
  const auto request = stage26_request(group);
  const std::vector<std::string> links = { "wrist3_link", "wrist2_link", "forearm_link", "upperarm_link",
                                           "shoulder_link", "base_link" };
  const std::vector<double> sides = { 0.015, 0.025, 0.04, 0.06, 0.10 };
  int trial = 0;
  for (const auto& pair : synthetic_joint_pairs()) {
    RobotState middle(model);
    std::vector<double> qmid(6, 0.0);
    for (int j = 0; j < 6; ++j)
      qmid[j] = 0.5 * (pair.first[j] + pair.second[j]);
    middle.setVariablePositions(qmid);
    middle.updateCollisionBodyTransforms();
    for (const auto& link : links) {
      const auto* link_model = model->getLinkModel(link);
      if (!link_model || link_model->getShapes().empty())
        continue;
      const Eigen::Vector3d center = middle.getCollisionBodyTransform(link, 0).translation();
      for (double side : sides) {
        ++trial;
        auto scene = synthetic_scene(model, center, side, "bullet");
        const auto& acm = scene->getAllowedCollisionMatrix();
        const auto env_ptr = scene->getCollisionEnv();
        const auto* env = env_ptr.get();
        RobotState start(model), end(model), midpoint(model);
        start.setVariablePositions(pair.first);
        end.setVariablePositions(pair.second);
        midpoint.setVariablePositions(qmid);
        start.updateCollisionBodyTransforms();
        end.updateCollisionBodyTransforms();
        midpoint.updateCollisionBodyTransforms();
        const Check start_check = collect(run_world_collision(env, request, start, acm));
        const Check end_check = collect(run_world_collision(env, request, end, acm));
        const Check middle_check = collect(run_world_collision(env, request, midpoint, acm));
        if (start_check.collision || end_check.collision || !middle_check.collision)
          continue;
        const Check ccd_check = collect(run_world_ccd(env, request, start, end, acm));
        out << "{\"test\":\"A\",\"trial\":" << trial << ",\"link\":";
        js(out, link);
        out << ",\"obstacle_center\":";
        e3(out, center);
        out << ",\"obstacle_side_m\":" << jn(side) << ",\"q0\":";
        write_double_array(out, pair.first);
        out << ",\"q1\":";
        write_double_array(out, pair.second);
        out << ",\"discrete_start_free\":" << (!start_check.collision ? "true" : "false")
            << ",\"discrete_end_free\":" << (!end_check.collision ? "true" : "false")
            << ",\"discrete_mid_collision\":" << (middle_check.collision ? "true" : "false")
            << ",\"continuous_collision\":" << (ccd_check.collision ? "true" : "false")
            << ",\"continuous_contact_fraction\":null,\"start_pairs\":";
        pairs(out, start_check.pairs);
        out << ",\"end_pairs\":";
        pairs(out, end_check.pairs);
        out << ",\"mid_pairs\":";
        pairs(out, middle_check.pairs);
        out << ",\"ccd_pairs\":";
        pairs(out, ccd_check.pairs);
        out << ",\"ccd_contacts\":";
        write_contact_json(out, ccd_check.contacts);
        out << "}\n";
        if (ccd_check.collision) {
          found_q0 = pair.first;
          found_q1 = pair.second;
          found_link = link;
          found_center = center;
          found_side = side;
          return true;
        }
      }
    }
  }
  out << "{\"test\":\"A\",\"status\":\"not_found\",\"trials\":" << trial << "}\n";
  return false;
}

static void run_positive_controls(const moveit::core::RobotModelConstPtr& model, const std::string& group,
                                  const fs::path& output_dir)
{
  std::ofstream out(output_dir / "stage26_positive_control_tests.jsonl");
  std::vector<double> q0, q1;
  std::string link;
  Eigen::Vector3d center(0.0, 0.0, 0.0);
  double side = 0.0;
  const bool found = find_swept_positive_control(model, group, out, q0, q1, link, center, side);
  if (!found) {
    out << "{\"test\":\"B\",\"status\":\"not_evaluable_positive_control_A_missing\"}\n";
    out << "{\"test\":\"C\",\"status\":\"not_evaluable_positive_control_A_missing\"}\n";
    return;
  }

  const auto request = stage26_request(group);
  auto free_scene = synthetic_scene(model, Eigen::Vector3d(100.0, 100.0, 100.0), side, "bullet");
  const auto& free_acm = free_scene->getAllowedCollisionMatrix();
  const auto free_env_ptr = free_scene->getCollisionEnv();
  const auto* free_env = free_env_ptr.get();
  RobotState free_start(model), free_end(model);
  free_start.setVariablePositions(q0);
  free_end.setVariablePositions(q1);
  free_start.updateCollisionBodyTransforms();
  free_end.updateCollisionBodyTransforms();
  const Check free_start_check = collect(run_world_collision(free_env, request, free_start, free_acm));
  const Check free_end_check = collect(run_world_collision(free_env, request, free_end, free_acm));
  const Check free_ccd_check = collect(run_world_ccd(free_env, request, free_start, free_end, free_acm));
  out << "{\"test\":\"B\",\"discrete_start_free\":" << (!free_start_check.collision ? "true" : "false")
      << ",\"discrete_end_free\":" << (!free_end_check.collision ? "true" : "false")
      << ",\"continuous_free\":" << (!free_ccd_check.collision ? "true" : "false")
      << ",\"obstacle_center\":[100.0,100.0,100.0],\"continuous_contact_fraction\":null}\n";

  auto acm_scene = synthetic_scene(model, center, side, "bullet");
  const auto formal_acm = acm_scene->getAllowedCollisionMatrix();
  auto allow_acm = formal_acm;
  allow_acm.setEntry(link, "stage26_synthetic_obstacle", true);
  const auto acm_env_ptr = acm_scene->getCollisionEnv();
  const auto* acm_env = acm_env_ptr.get();
  RobotState middle(model);
  std::vector<double> qmid(6, 0.0);
  for (int j = 0; j < 6; ++j)
    qmid[j] = 0.5 * (q0[j] + q1[j]);
  middle.setVariablePositions(qmid);
  middle.updateCollisionBodyTransforms();
  const Check formal_check = collect(run_world_collision(acm_env, request, middle, formal_acm));
  const Check allowed_check = collect(run_world_collision(acm_env, request, middle, allow_acm));
  out << "{\"test\":\"C\",\"link\":";
  js(out, link);
  out << ",\"world_object\":\"stage26_synthetic_obstacle\",\"formal_acm_collision\":"
      << (formal_check.collision ? "true" : "false") << ",\"modified_acm_collision\":"
      << (allowed_check.collision ? "true" : "false") << ",\"formal_acm_modified_for_formal_run\":false}\n";
}

static void write_provenance(const fs::path& output_dir, const std::string& backend,
                             const planning_scene::PlanningScene& scene, const std::string& api,
                             const std::string& status)
{
  std::ofstream out(output_dir / "stage26_runtime_provenance.json");
  out << "{\"status\":";
  js(out, status);
  out << ",\"backend\":";
  js(out, backend);
  out << ",\"active_detector_name\":";
  js(out, scene.getCollisionDetectorName());
  out << ",\"collision_environment_concrete_class\":";
  js(out, demangle(typeid(*scene.getCollisionEnv()).name()));
  out << ",\"api_call\":";
  js(out, api);
  out << ",\"request_configuration\":{\"contacts\":true,\"max_contacts\":1,\"max_contacts_per_pair\":1,\"pad_environment_collisions\":true,\"pad_self_collisions\":false,\"distance\":false,\"acm_modified\":false}";
  out << ",\"world_object_ids\":[";
  const auto ids = scene.getWorld()->getObjectIds();
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i)
      out << ',';
    js(out, ids[i]);
  }
  out << "]";
  if (backend == "bullet") {
    out << ",\"bullet_active_links\":[";
    const auto& links = scene.getRobotModel()->getLinkModels();
    for (size_t i = 0; i < links.size(); ++i) {
        if (i)
          out << ',';
        js(out, links[i]->getName());
    }
    out << "]";
  }
  out << ",\"continuous_contact_fraction_available\":false,\"continuous_contact_fraction\":null"
      << ",\"continuous_self_collision_native\":false}\n";
}

static int run_probe(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
      "stage26_continuous_probe", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  std::string interval_csv, parts_dir, output_dir, backend, group;
  int run = 1;
  bool capability_only = false;
  bool positive_controls = true;
  node->get_parameter_or("interval_csv", interval_csv, std::string{});
  node->get_parameter_or("parts_dir", parts_dir, std::string{});
  node->get_parameter_or("output_dir", output_dir, std::string{});
  node->get_parameter_or("backend", backend, std::string("bullet"));
  node->get_parameter_or("run_index", run, 1);
  node->get_parameter_or("capability_only", capability_only, false);
  node->get_parameter_or("positive_controls", positive_controls, true);
  node->get_parameter_or("group_name", group, std::string("fairino5_v6_group"));
  if (interval_csv.empty() || parts_dir.empty() || output_dir.empty())
    return 2;
  const char* shape_mode = std::getenv("FR5_BULLET_SHAPE_MODE");
  if (shape_mode == nullptr || std::string(shape_mode) != "use_shape_type") {
    std::cerr << "FR5_BULLET_SHAPE_MODE must equal use_shape_type for formal Stage 2.6 checks\n";
    rclcpp::shutdown();
    return 10;
  }

  fs::create_directories(output_dir);
  const fs::path output_path(output_dir);
  const auto intervals = read_intervals(interval_csv);
  const auto meshes = read_parts(parts_dir);
  robot_model_loader::RobotModelLoader::Options options("robot_description");
  options.load_kinematics_solvers = false;
  robot_model_loader::RobotModelLoader loader(node, options);
  auto model = loader.getModel();
  if (!model)
    return 3;
  auto scene = std::make_unique<planning_scene::PlanningScene>(model);
  scene->processCollisionObjectMsg(world_object(meshes));
  if (backend == "bullet")
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create());
  else
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  if (!model->getJointModelGroup(group))
    return 4;
  const auto request = stage26_request(group);
  const auto& acm = scene->getAllowedCollisionMatrix();
  write_provenance(output_dir, backend, *scene,
                   backend == "bullet" ? "CollisionEnvBullet::checkRobotCollision(req,res,state1,state2,acm)"
                                        : "CollisionEnvFCL::checkRobotCollision(req,res,state1,state2,acm)",
                   capability_only ? "capability_probe" : "native_runtime_observed");

  if (backend == "fcl" && capability_only) {
    std::ofstream out(output_path / "stage26_fcl_capability_probe.json");
    if (intervals.empty()) {
      out << "{\"available\":false,\"status\":\"not_available\",\"reason\":\"no_interval_input\"}\n";
      rclcpp::shutdown();
      return 0;
    }
    RobotState state0(model), state1(model);
    state0.setVariablePositions(intervals.front().q0);
    state1.setVariablePositions(intervals.front().q1);
    state0.updateCollisionBodyTransforms();
    state1.updateCollisionBodyTransforms();
    collision_detection::CollisionResult result = run_world_ccd(scene->getCollisionEnv().get(), request, state0, state1, acm);
    out << "{\"available\":false,\"status\":\"not_available\",\"reason\":\"MoveIt_FCL_backend_does_not_implement_continuous_collision\",\"api_invoked\":true,\"result_collision\":"
        << (result.collision ? "true" : "false") << "}\n";
    rclcpp::shutdown();
    return 0;
  }

  if (backend != "bullet") {
    rclcpp::shutdown();
    return 11;
  }
  std::ofstream interval_out(output_path / "stage26_continuous_robot_world_intervals.jsonl");
  std::ofstream self_out(output_path / "stage26_self_collision_states.jsonl");
  const auto env_ptr = scene->getCollisionEnv();
  const auto* env = env_ptr.get();
  std::size_t collision_count = 0;
  std::size_t self_collision_count = 0;
  RobotState state0(model), state1(model);
  for (const auto& interval : intervals) {
    state0.setVariablePositions(interval.q0);
    state1.setVariablePositions(interval.q1);
    state0.updateCollisionBodyTransforms();
    state1.updateCollisionBodyTransforms();
    const Check continuous = collect(run_world_ccd(env, request, state0, state1, acm));
    const Check self0 = collect(run_self_collision(*scene, request, state0, acm));
    if (continuous.collision)
      ++collision_count;
    if (self0.collision)
      ++self_collision_count;
    self_out << "{\"state_index\":" << interval.index << ",\"time_s\":" << jn(interval.t0)
             << ",\"collision\":" << (self0.collision ? "true" : "false")
             << ",\"start_pairs\":";
    pairs(self_out, self0.pairs);
    self_out << ",\"end_pairs\":[]";
    self_out << "}\n";
    interval_out << "{\"interval_index\":" << interval.index << ",\"time_start_s\":" << jn(interval.t0)
                 << ",\"time_end_s\":" << jn(interval.t1) << ",\"duration_s\":" << jn(interval.t1 - interval.t0)
                 << ",\"process_order_index\":";
    write_string_or_null(interval_out, interval.process_order);
    interval_out << ",\"spray_state\":";
    write_string_or_null(interval_out, interval.spray_state);
    interval_out << ",\"process_kind\":";
    write_string_or_null(interval_out, interval.process_kind);
    interval_out << ",\"segment_id\":";
    write_string_or_null(interval_out, interval.segment_id);
    interval_out << ",\"transition_id\":";
    write_string_or_null(interval_out, interval.transition_id);
    interval_out << ",\"source_boundary\":";
    write_string_or_null(interval_out, interval.source_boundary);
    interval_out << ",\"q_start\":";
    write_double_array(interval_out, interval.q0);
    interval_out << ",\"q_end\":";
    write_double_array(interval_out, interval.q1);
    interval_out << ",\"discrete_start_free\":null,\"discrete_end_free\":null"
                 << ",\"continuous_collision\":" << (continuous.collision ? "true" : "false")
                 << ",\"continuous_contact_fraction\":null,\"collision_method\":\"native_bullet_robot_world_ccd\",\"ccd_api_called\":true,\"first_contact\":";
    if (continuous.contacts.empty())
      interval_out << "null";
    else
      write_contact_json(interval_out, { continuous.contacts.front() });
    interval_out << ",\"discrete_start_pairs\":[],\"discrete_end_pairs\":[]";
    interval_out << ",\"continuous_pairs\":";
    pairs(interval_out, continuous.pairs);
    interval_out << ",\"continuous_contacts\":";
    write_contact_json(interval_out, continuous.contacts);
    interval_out << "}\n";
  }
  if (!intervals.empty()) {
    const auto& last = intervals.back();
    const Check final_self = collect(run_self_collision(*scene, request, state1, acm));
    if (final_self.collision)
      ++self_collision_count;
    self_out << "{\"state_index\":" << (last.index + 1) << ",\"time_s\":" << jn(last.t1)
             << ",\"collision\":" << (final_self.collision ? "true" : "false")
             << ",\"start_pairs\":";
    pairs(self_out, final_self.pairs);
    self_out << ",\"end_pairs\":[]}\n";
  }
  if (positive_controls && run == 1)
    run_positive_controls(model, group, output_path);
  std::ofstream summary(output_path / "stage26_native_summary.json");
  summary << "{\"backend\":\"bullet\",\"run_index\":" << run << ",\"interval_count\":" << intervals.size()
          << ",\"intervals_with_continuous_collision\":" << collision_count
          << ",\"states_with_endpoint_self_collision\":" << self_collision_count
          << ",\"all_intervals_executed\":true,\"skipped_intervals\":0,\"unchecked_intervals\":0,\"continuous_contact_fraction_available\":false}\n";
  rclcpp::shutdown();
  return 0;
}

#ifndef STAGE26R_EMBEDDED
int main(int argc, char** argv)
{
  try {
    return run_probe(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "stage26_continuous_probe fatal: " << error.what() << '\n';
    if (rclcpp::ok())
      rclcpp::shutdown();
    return 20;
  }
}
#endif
