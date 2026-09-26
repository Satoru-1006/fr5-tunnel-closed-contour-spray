// Compact formal Stage 2.3B PlanningScene runner.
// It reuses the existing native input/scene helpers but omits the diagnostic
// manifold dump so a triangle-compound representation can be rebaselined at
// full frozen-candidate scale without changing the collision request.

#define main stage23b_original_probe_main
#include "../stage23a7/stage23a7_runtime_audit_probe.cpp"
#undef main
#ifdef checkSelfCollision
#undef checkSelfCollision
#endif

static double minimum_depth(const Check& c)
{
  if (c.contacts.empty())
    return std::numeric_limits<double>::quiet_NaN();
  double value = std::numeric_limits<double>::infinity();
  for (const auto& contact : c.contacts)
    value = std::min(value, contact.depth);
  return value;
}

static bool has_self_pair(const Check& c)
{
  for (const auto& pair : c.pairs)
    if (pair.find("horseshoe_collision_compound") == std::string::npos)
      return true;
  return false;
}

static bool has_world_pair(const Check& c)
{
  for (const auto& pair : c.pairs)
    if (pair.find("horseshoe_collision_compound") != std::string::npos)
      return true;
  return false;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
      "stage23b_formal_probe", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  std::string candidate_csv, parts_dir, output_dir, backend, group;
  int run = 1;
  node->get_parameter_or("candidate_csv", candidate_csv, std::string{});
  node->get_parameter_or("parts_dir", parts_dir, std::string{});
  node->get_parameter_or("output_dir", output_dir, std::string{});
  node->get_parameter_or("backend", backend, std::string("bullet"));
  node->get_parameter_or("run_index", run, 1);
  node->get_parameter_or("group_name", group, std::string("fairino5_v6_group"));
  if (candidate_csv.empty() || parts_dir.empty() || output_dir.empty())
    return 2;

  fs::create_directories(output_dir);
  const auto candidates = read_candidates(candidate_csv);
  const auto meshes = read_parts(parts_dir);
  robot_model_loader::RobotModelLoader::Options opt("robot_description");
  opt.load_kinematics_solvers = false;
  robot_model_loader::RobotModelLoader loader(node, opt);
  auto model = loader.getModel();
  if (!model)
    return 3;
  auto scene = std::make_unique<planning_scene::PlanningScene>(model);
  scene->processCollisionObjectMsg(world_object(meshes));
  if (backend == "bullet")
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create());
  else
    scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  auto* jmg = model->getJointModelGroup(group);
  if (!jmg)
    return 4;

  if (backend == "bullet" && run == 1)
    dump_shapes(*scene, backend, output_dir, model);
  auto* bullet_env = dynamic_cast<collision_detection::CollisionEnvBullet*>(scene->getCollisionEnvNonConst().get());
  std::ofstream full(output_dir + "/" + backend + "_runtime_results_run" + std::to_string(run) + ".jsonl");
  std::ofstream provenance(output_dir + "/runtime_provenance_run" + std::to_string(run) + ".json");
  std::ifstream maps("/proc/self/maps");
  std::ofstream mapout(output_dir + "/proc_self_maps_run" + std::to_string(run) + ".txt");
  mapout << maps.rdbuf();
  provenance << "{\"status\":\"native_runtime_observed\",\"backend\":";
  js(provenance, backend);
  provenance << ",\"active_detector_name\":";
  js(provenance, scene->getCollisionDetectorName());
  provenance << ",\"collision_environment_concrete_class\":";
  js(provenance, demangle(typeid(*scene->getCollisionEnv()).name()));
  provenance << ",\"collision_detector_allocator\":";
  js(provenance, demangle(typeid(*scene->getCollisionEnv()).name()));
  provenance << ",\"sizeof_btScalar\":" << sizeof(btScalar)
             << ",\"bt_use_double_precision\":" << (sizeof(btScalar) == 8 ? "true" : "false")
             << ",\"compiler\":\"gcc_runtime\",\"build_type\":\"O2\",\"ros_distro\":\"jazzy\","
                "\"moveit_version\":\"2.12.4\",\"bullet_version\":\"3.24\","
                "\"formal_acm_modified\":false}\n";

  const auto request = req(group, true);
  for (const auto& candidate : candidates)
  {
    RobotState state(model);
    state.setVariablePositions(candidate.q);
    state.updateCollisionBodyTransforms();
    collision_detection::CollisionResult result;
    result.clear();
    // Formal node validity includes both robot self-collision and
    // robot-world collision.  The older diagnostic probe intentionally used
    // a self-only call; Stage 2.3B must use the full PlanningScene API.
    scene->checkCollision(request, result, state);
    const auto check = collect(result);
    full << "{\"candidate_id\":";
    js(full, candidate.id);
    full << ",\"waypoint_id\":" << candidate.waypoint << ",\"joint_values\":[";
    for (std::size_t i = 0; i < candidate.q.size(); ++i)
    {
      if (i)
        full << ',';
      full << jn(candidate.q[i]);
    }
    full << "],\"backend\":";
    js(full, backend);
    full << ",\"valid\":" << (check.collision ? "false" : "true")
         << ",\"self_collision\":" << (has_self_pair(check) ? "true" : "false")
         << ",\"robot_world_collision\":" << (has_world_pair(check) ? "true" : "false")
         << ",\"collision_pairs\":";
    pairs(full, check.pairs);
    full << ",\"raw_contact_distance\":";
    const double depth = minimum_depth(check);
    if (std::isnan(depth))
      full << "null";
    else
      full << jn(depth);
    // CollisionResult contact depth is not a clearance query.  The formal
    // Stage 2.3B interface has no certified clearance backend, so preserve the
    // raw contact value above and keep minimum_distance explicitly unavailable.
    full << ",\"minimum_distance\":null,\"minimum_distance_status\":\"not_available\"";
    full << ",\"request_contacts\":true,\"collision_body_update_called\":true}\n";
  }

  std::ofstream summary(output_dir + "/" + backend + "_runtime_summary_run" + std::to_string(run) + ".json");
  summary << "{\"backend\":";
  js(summary, backend);
  summary << ",\"run_index\":" << run << ",\"candidate_count\":" << candidates.size()
          << ",\"active_detector_name\":";
  js(summary, scene->getCollisionDetectorName());
  summary << ",\"formal_acm_modified\":false,\"bullet_environment_observed\":"
          << (bullet_env != nullptr ? "true" : "false") << "}\n";
  rclcpp::shutdown();
  return 0;
}
