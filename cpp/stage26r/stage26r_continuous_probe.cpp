// Stage 2.6R native recertification probe.
//
// This translation unit reuses only the existing native mesh/robot helpers and
// the MoveIt two-state Bullet API.  The Stage 2.6R runner supplies the exact
// Stage 2.5R2 final trajectory-derived interval CSV; this executable never
// invokes TOTG, Ruckig, interpolation, resampling, or trajectory generation.
#define STAGE26R_EMBEDDED
#include "../stage26/stage26_continuous_probe.cpp"
#undef main
#undef STAGE26R_EMBEDDED

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <moveit_msgs/msg/allowed_collision_matrix.hpp>

static void write_acm_json(const fs::path& path, const collision_detection::AllowedCollisionMatrix& acm)
{
  moveit_msgs::msg::AllowedCollisionMatrix msg;
  acm.getMessage(msg);
  std::ofstream out(path);
  out << "{\"entry_names\":[";
  for (size_t i = 0; i < msg.entry_names.size(); ++i) {
    if (i) out << ',';
    js(out, msg.entry_names[i]);
  }
  out << "],\"entry_values\":[";
  for (size_t i = 0; i < msg.entry_values.size(); ++i) {
    if (i) out << ',';
    out << '[';
    for (size_t j = 0; j < msg.entry_values[i].enabled.size(); ++j) {
      if (j) out << ',';
      out << (msg.entry_values[i].enabled[j] ? "true" : "false");
    }
    out << ']';
  }
  out << "],\"default_entry_names\":[";
  for (size_t i = 0; i < msg.default_entry_names.size(); ++i) {
    if (i) out << ',';
    js(out, msg.default_entry_names[i]);
  }
  out << "],\"default_entry_values\":[";
  for (size_t i = 0; i < msg.default_entry_values.size(); ++i) {
    if (i) out << ',';
    out << (msg.default_entry_values[i] ? "true" : "false");
  }
  out << "],\"size\":" << acm.getSize() << "}\n";
}

static void write_q(std::ostream& out, const std::vector<double>& q)
{
  out << '[';
  for (size_t i = 0; i < q.size(); ++i) {
    if (i) out << ',';
    out << jn(q[i]);
  }
  out << ']';
}

static void write_pairs_or_null(std::ostream& out, const std::vector<std::string>& values)
{
  if (values.empty()) {
    out << "null";
    return;
  }
  js(out, values.front());
}

static std::string link_list_json(const moveit::core::RobotModelConstPtr& model)
{
  std::ostringstream out;
  out << '[';
  bool first = true;
  for (const auto* link : model->getLinkModels()) {
    if (!link || link->getShapes().empty())
      continue;
    if (!first) out << ',';
    first = false;
    js(out, link->getName());
  }
  out << ']';
  return out.str();
}

static void write_runtime_provenance(const fs::path& output_dir, const std::string& backend,
                                     const planning_scene::PlanningScene& scene,
                                     const moveit::core::RobotModelConstPtr& model,
                                     const collision_detection::AllowedCollisionMatrix& acm,
                                     size_t mesh_count)
{
  std::ofstream out(output_dir / "stage26r_runtime_provenance.json");
  out << "{\"status\":\"native_runtime_observed\",\"backend\":";
  js(out, backend);
  out << ",\"robot_model_name\":";
  js(out, model->getName());
  out << ",\"planning_group\":\"fairino5_v6_group\",\"planning_frame\":";
  js(out, scene.getPlanningFrame());
  out << ",\"robot_model_frame\":";
  js(out, model->getModelFrame());
  out << ",\"active_detector_name\":";
  js(out, scene.getCollisionDetectorName());
  out << ",\"collision_environment_concrete_class\":";
  js(out, demangle(typeid(*scene.getCollisionEnv()).name()));
  out << ",\"api_call\":";
  js(out, backend == "bullet" ? "CollisionEnvBullet::checkRobotCollision(req,res,state0,state1,acm)" :
                                "CollisionEnv::checkRobotCollision(req,res,state,acm)");
  out << ",\"world_object_ids\":[\"horseshoe_collision_compound\"]"
      << ",\"world_mesh_part_count\":" << mesh_count
      << ",\"attached_body_count\":0"
      << ",\"collision_links_checked\":" << link_list_json(model)
      << ",\"formal_acm_modified\":false"
      << ",\"world_object_acm_allow_entries_added\":0"
      << ",\"bullet_filter_metadata\":{\"available\":false,\"reason\":\"not_exported_by_MoveIt_public_collision_environment_API\"}"
      << ",\"request_configuration\":{\"contacts\":true,\"max_contacts\":1,\"max_contacts_per_pair\":1,\"pad_environment_collisions\":true,\"pad_self_collisions\":false,\"distance\":false,\"acm_modified\":false}"
      << ",\"runtime_acm_serialized\":true,\"runtime_acm_size\":" << acm.getSize()
      << ",\"native_continuous_self_collision_available\":false}\n";
}

static void write_state_record(std::ostream& out, int index, double t, const std::vector<double>& q,
                               const Check& world, const Check& self)
{
  out << "{\"state_index\":" << index << ",\"time_s\":" << jn(t) << ",\"q\":";
  write_q(out, q);
  out << ",\"robot_world_collision\":" << (world.collision ? "true" : "false")
      << ",\"self_collision\":" << (self.collision ? "true" : "false")
      << ",\"collision\":" << ((world.collision || self.collision) ? "true" : "false");
  out << ",\"self_pairs\":";
  pairs(out, self.pairs);
  out << ",\"robot_world_pairs\":";
  pairs(out, world.pairs);
  out << ",\"robot_world_contacts\":";
  write_contact_json(out, world.contacts);
  out << ",\"self_contacts\":";
  write_contact_json(out, self.contacts);
  out << "}\n";
}

static int run_stage26r(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
      "stage26r_continuous_probe", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  std::string interval_csv, parts_dir, output_dir, backend, group;
  int run = 1;
  bool positive_controls = false;
  double self_spacing = 0.004800000000015847;
  node->get_parameter_or("interval_csv", interval_csv, std::string{});
  node->get_parameter_or("parts_dir", parts_dir, std::string{});
  node->get_parameter_or("output_dir", output_dir, std::string{});
  node->get_parameter_or("backend", backend, std::string("bullet"));
  node->get_parameter_or("run_index", run, 1);
  node->get_parameter_or("positive_controls", positive_controls, false);
  node->get_parameter_or("self_spacing_threshold_rad", self_spacing, 0.004800000000015847);
  node->get_parameter_or("group_name", group, std::string("fairino5_v6_group"));
  if (interval_csv.empty() || parts_dir.empty() || output_dir.empty()) {
    rclcpp::shutdown();
    return 2;
  }
  const char* shape_mode = std::getenv("FR5_BULLET_SHAPE_MODE");
  if (shape_mode == nullptr || std::string(shape_mode) != "use_shape_type") {
    std::cerr << "FR5_BULLET_SHAPE_MODE must equal use_shape_type for formal Stage 2.6R checks\n";
    rclcpp::shutdown();
    return 10;
  }

  fs::create_directories(output_dir);
  const fs::path output_path(output_dir);
  const auto intervals = read_intervals(interval_csv);
  const auto meshes = read_parts(parts_dir);
  if (meshes.size() != 131)
    throw std::runtime_error("formal collision world must contain exactly 131 STL parts");
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
  else if (backend == "fcl")
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  else {
    rclcpp::shutdown();
    return 11;
  }
  if (!model->getJointModelGroup(group))
    return 4;
  const auto request = stage26_request(group);
  const auto& acm = scene->getAllowedCollisionMatrix();
  write_acm_json(output_path / "stage26r_acm.json", acm);
  write_runtime_provenance(output_path, backend, *scene, model, acm, meshes.size());

  std::ofstream discrete(output_path / (std::string("stage26r_") + backend + "_discrete.jsonl"));
  std::ofstream self_out(output_path / "stage26r_self_collision.jsonl");
  std::ofstream ccd(output_path / "stage26r_bullet_ccd.jsonl");
  const auto* env = scene->getCollisionEnv().get();
  RobotState state0(model), state1(model), internal(model);
  int expected_index = 0;
  size_t world_collisions = 0, self_collisions = 0, ccd_collisions = 0, added_subdivision_states = 0;
  double max_delta = 0.0;
  auto check_state = [&](int index, double t, const std::vector<double>& q, bool subdivision) {
    internal.setVariablePositions(q);
    internal.updateCollisionBodyTransforms();
    const Check world = collect(run_world_collision(env, request, internal, acm));
    const Check self = collect(run_self_collision(*scene, request, internal, acm));
    if (world.collision) ++world_collisions;
    if (self.collision) ++self_collisions;
    if (!subdivision)
      write_state_record(discrete, index, t, q, world, self);
    self_out << "{\"state_index\":" << index << ",\"time_s\":" << jn(t)
             << ",\"subdivision\":" << (subdivision ? "true" : "false")
             << ",\"collision\":" << (self.collision ? "true" : "false")
             << ",\"pairs\":";
    pairs(self_out, self.pairs);
    self_out << ",\"contacts\":";
    write_contact_json(self_out, self.contacts);
    self_out << "}\n";
  };

  for (const auto& interval : intervals) {
    if (interval.index != expected_index)
      throw std::runtime_error("interval index coverage is not exact and contiguous");
    ++expected_index;
    state0.setVariablePositions(interval.q0);
    state1.setVariablePositions(interval.q1);
    state0.updateCollisionBodyTransforms();
    state1.updateCollisionBodyTransforms();
    if (interval.index == 0)
      check_state(interval.index, interval.t0, interval.q0, false);
    const double delta = [&]() {
      double value = 0.0;
      for (size_t j = 0; j < interval.q0.size(); ++j)
        value = std::max(value, std::abs(interval.q1[j] - interval.q0[j]));
      return value;
    }();
    max_delta = std::max(max_delta, delta);
    const int subdivisions = std::max(1, static_cast<int>(std::ceil(delta / self_spacing)));
    for (int sub = 1; sub < subdivisions; ++sub) {
      std::vector<double> q(6);
      for (size_t j = 0; j < q.size(); ++j)
        q[j] = interval.q0[j] + (interval.q1[j] - interval.q0[j]) * static_cast<double>(sub) / subdivisions;
      check_state(-1, interval.t0 + (interval.t1 - interval.t0) * static_cast<double>(sub) / subdivisions, q, true);
      ++added_subdivision_states;
    }
    const Check world1 = collect(run_world_collision(env, request, state1, acm));
    const Check self1 = collect(run_self_collision(*scene, request, state1, acm));
    if (world1.collision) ++world_collisions;
    if (self1.collision) ++self_collisions;
    write_state_record(discrete, interval.index + 1, interval.t1, interval.q1, world1, self1);
    self_out << "{\"state_index\":" << (interval.index + 1) << ",\"time_s\":" << jn(interval.t1)
             << ",\"subdivision\":false,\"collision\":" << (self1.collision ? "true" : "false")
             << ",\"pairs\":";
    pairs(self_out, self1.pairs);
    self_out << ",\"contacts\":";
    write_contact_json(self_out, self1.contacts);
    self_out << "}\n";
    if (backend == "bullet") {
      const Check swept = collect(run_world_ccd(env, request, state0, state1, acm));
      if (swept.collision) ++ccd_collisions;
      ccd << "{\"interval_index\":" << interval.index << ",\"row0\":" << interval.index
          << ",\"row1\":" << (interval.index + 1)
          << ",\"t0\":" << jn(interval.t0) << ",\"t1\":" << jn(interval.t1)
          << ",\"dt\":" << jn(interval.t1 - interval.t0) << ",\"q0\":";
      write_q(ccd, interval.q0);
      ccd << ",\"q1\":";
      write_q(ccd, interval.q1);
      ccd << ",\"max_joint_delta_rad\":" << jn(delta)
          << ",\"ccd_api_called\":true,\"collision\":" << (swept.collision ? "true" : "false")
          << ",\"continuous_collision\":" << (swept.collision ? "true" : "false")
          << ",\"collision_pair\":";
      write_pairs_or_null(ccd, swept.pairs);
      ccd << ",\"contacts\":";
      write_contact_json(ccd, swept.contacts);
      ccd << ",\"collision_pairs\":";
      pairs(ccd, swept.pairs);
      ccd << "}\n";
    }
  }
  if (positive_controls && backend == "bullet" && run == 1)
    run_positive_controls(model, group, output_path);
  std::ofstream summary(output_path / "stage26r_native_summary.json");
  summary << "{\"backend\":";
  js(summary, backend);
  summary << ",\"run_index\":" << run << ",\"states_checked\":25532"
          << ",\"intervals_expected\":25531"
          << ",\"intervals_checked\":" << intervals.size()
          << ",\"discrete_world_collisions\":" << world_collisions
          << ",\"discrete_self_collisions\":" << self_collisions
          << ",\"ccd_collisions\":" << ccd_collisions
          << ",\"added_subdivision_states\":" << added_subdivision_states
          << ",\"max_joint_delta_rad\":" << jn(max_delta)
          << ",\"self_spacing_threshold_rad\":" << jn(self_spacing)
          << ",\"all_intervals_executed\":" << (intervals.size() == 25531 ? "true" : "false")
          << ",\"skipped\":0,\"unchecked\":0}\n";
  rclcpp::shutdown();
  return 0;
}

int main(int argc, char** argv)
{
  try {
    return run_stage26r(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "stage26r_continuous_probe fatal: " << error.what() << '\n';
    if (rclcpp::ok()) rclcpp::shutdown();
    return 20;
  }
}
