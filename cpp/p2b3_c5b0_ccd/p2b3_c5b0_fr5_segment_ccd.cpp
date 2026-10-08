// Bounded real-FR5 C5B0 hotfix probe.
// The C5A native loader and C4 box-scene construction are reused verbatim.
#define main p2b3_c5a_native_main
#include "../p2b3_c5a_native/p2b3_c5a_native.cpp"
#undef main

#include <moveit/collision_detection_bullet/collision_detector_allocator_bullet.hpp>

namespace {

struct HotfixArgs {
  std::string urdf, srdf, scene, trajectory, output;
  std::string group{"fairino5_v6_group"};
  int start{-1};
  int end{-1};
};

HotfixArgs parse_hotfix_args(int argc, char** argv) {
  HotfixArgs args;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key(argv[i]), value(argv[i + 1]);
    if (key == "--urdf") args.urdf = value;
    else if (key == "--srdf") args.srdf = value;
    else if (key == "--scene") args.scene = value;
    else if (key == "--trajectory") args.trajectory = value;
    else if (key == "--output") args.output = value;
    else if (key == "--group") args.group = value;
    else if (key == "--start") args.start = std::stoi(value);
    else if (key == "--end") args.end = std::stoi(value);
    else throw std::runtime_error("unknown_argument:" + key);
  }
  if (args.urdf.empty() || args.srdf.empty() || args.scene.empty() ||
      args.trajectory.empty() || args.output.empty() || args.start < 0 ||
      args.end <= args.start)
    throw std::runtime_error("required_arguments:--urdf --srdf --scene --trajectory --output --start --end");
  return args;
}

std::unique_ptr<planning_scene::PlanningScene> make_bullet_scene(
    const moveit::core::RobotModelConstPtr& model, const fs::path& path) {
  auto scene = std::make_unique<planning_scene::PlanningScene>(model);
  const auto objects = read_scene_csv(path);
  for (const auto& object : objects)
    if (!scene->processCollisionObjectMsg(box_message(object)))
      throw std::runtime_error("scene_rejected_object:" + object.id);
  if (scene->getWorld()->getObjectIds().size() != 181)
    throw std::runtime_error("installed_scene_object_count_mismatch");
  scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create());
  return scene;
}

collision_detection::CollisionResult discrete_result(
    const collision_detection::CollisionEnv* environment,
    const collision_detection::CollisionRequest& request,
    const moveit::core::RobotState& state,
    const collision_detection::AllowedCollisionMatrix& acm) {
  collision_detection::CollisionResult result;
  environment->checkRobotCollision(request, result, state, acm);
  return result;
}

void write_output(const fs::path& output, const std::string& json) {
  std::ofstream stream(output);
  if (!stream) throw std::runtime_error("cannot_write_output:" + output.string());
  stream << json << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const HotfixArgs args = parse_hotfix_args(argc, argv);
    if (!rclcpp::ok()) rclcpp::init(argc, argv);
    auto urdf = std::make_shared<urdf::Model>();
    if (!urdf->initFile(args.urdf)) throw std::runtime_error("URDF_parse_failed");
    auto srdf = std::make_shared<srdf::Model>();
    if (!srdf->initFile(*urdf, args.srdf)) throw std::runtime_error("SRDF_parse_failed");
    auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
    const auto group = model->getJointModelGroup(args.group);
    if (!group) throw std::runtime_error("robot_group_missing:" + args.group);
    const std::set<std::string> expected_collision_links{
      "base_link", "shoulder_link", "upperarm_link", "forearm_link",
      "wrist1_link", "wrist2_link", "wrist3_link"};
    std::set<std::string> actual_collision_links;
    for (const auto* link : model->getLinkModelsWithCollisionGeometry())
      actual_collision_links.insert(link->getName());
    if (actual_collision_links != expected_collision_links)
      throw std::runtime_error("fr5_collision_geometry_link_set_mismatch:expected=7:actual=" +
                               std::to_string(actual_collision_links.size()));
    const auto rows = read_joint_csv(args.trajectory);
    if (args.end >= static_cast<int>(rows.size())) throw std::runtime_error("segment_index_out_of_range");
    auto scene = make_bullet_scene(model, args.scene);
    const auto& acm = scene->getAllowedCollisionMatrix();
    const auto* environment = scene->getCollisionEnv().get();
    if (!environment) throw std::runtime_error("bullet_collision_environment_unavailable");
    const auto request = collision_request(args.group);
    RobotState state0(model), state1(model);
    set_state(state0, group, rows[static_cast<std::size_t>(args.start)].q);
    set_state(state1, group, rows[static_cast<std::size_t>(args.end)].q);
    const auto endpoint_a = discrete_result(environment, request, state0, acm);
    const auto endpoint_b = discrete_result(environment, request, state1, acm);

    double max_delta = 0.0;
    for (std::size_t joint = 0; joint < kDofs; ++joint)
      max_delta = std::max(max_delta, std::abs(rows[static_cast<std::size_t>(args.end)].q[joint] -
                                               rows[static_cast<std::size_t>(args.start)].q[joint]));
    const int steps = std::max(1, static_cast<int>(std::ceil(max_delta / kMaxStep)));
    int discrete_collision_samples = 0;
    int discrete_intermediate_collision_samples = 0;
    for (int step = 0; step <= steps; ++step) {
      const double fraction = static_cast<double>(step) / static_cast<double>(steps);
      std::array<double, kDofs> q{};
      for (std::size_t joint = 0; joint < kDofs; ++joint)
        q[joint] = rows[static_cast<std::size_t>(args.start)].q[joint] +
                   fraction * (rows[static_cast<std::size_t>(args.end)].q[joint] -
                               rows[static_cast<std::size_t>(args.start)].q[joint]);
      RobotState sample(model);
      set_state(sample, group, q);
      const auto result = discrete_result(environment, request, sample, acm);
      if (result.collision) {
        ++discrete_collision_samples;
        if (step > 0 && step < steps) ++discrete_intermediate_collision_samples;
      }
    }

    collision_detection::CollisionResult continuous;
    environment->checkRobotCollision(request, continuous, state0, state1, acm);
    std::ostringstream json;
    json << "{\"schema\":\"p2b3-c5b0-fr5-segment-ccd-v1\""
         << ",\"status\":\"AVAILABLE\""
         << ",\"robot\":\"FAIRINO_FR5\""
         << ",\"scene_object_count\":181"
         << ",\"scene_source\":\"C4_authoritative_scene_manifest_boxes\""
         << ",\"group\":" << json_quote(args.group)
         << ",\"segment_start_waypoint\":" << args.start
         << ",\"segment_end_waypoint\":" << args.end
         << ",\"time_start_s\":" << number(rows[static_cast<std::size_t>(args.start)].time)
         << ",\"time_end_s\":" << number(rows[static_cast<std::size_t>(args.end)].time)
         << ",\"q_start\":[";
    for (std::size_t joint = 0; joint < kDofs; ++joint) { if (joint) json << ','; json << number(rows[static_cast<std::size_t>(args.start)].q[joint]); }
    json << "],\"q_end\":[";
    for (std::size_t joint = 0; joint < kDofs; ++joint) { if (joint) json << ','; json << number(rows[static_cast<std::size_t>(args.end)].q[joint]); }
    json << "]"
         << ",\"collision_backend\":\"MoveIt2 CollisionEnvBullet\""
         << ",\"collision_api\":\"CollisionEnvBullet::checkRobotCollision(req,res,state1,state2,acm)\""
         << ",\"collision_request\":{\"pad_environment_collisions\":true,\"pad_self_collisions\":false,\"acm_applied\":true}"
         << ",\"endpoint_a_collision\":" << (endpoint_a.collision ? "true" : "false")
         << ",\"endpoint_b_collision\":" << (endpoint_b.collision ? "true" : "false")
         << ",\"adaptive_discrete_interpolation\":{\"steps\":" << steps
         << ",\"sample_count\":" << (steps + 1)
         << ",\"collision_sample_count\":" << discrete_collision_samples
         << ",\"intermediate_collision_sample_count\":" << discrete_intermediate_collision_samples << "}"
         << ",\"robot_world_ccd_collision\":" << (continuous.collision ? "true" : "false")
         << ",\"robot_world_ccd_pairs\":" << contact_pairs(continuous)
         << ",\"strict_self_ccd\":null"
         << ",\"strict_self_ccd_status\":\"NOT_AVAILABLE\""
         << ",\"clearance_m\":null"
         << ",\"hardware_validation\":null"
         << ",\"hardware_validation_status\":\"NOT_AVAILABLE\""
         << "}";
    const std::string result = json.str();
    write_output(args.output, result);
    std::cout << result << '\n';
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "p2b3_c5b0_fr5_segment_ccd: " << error.what() << '\n';
    if (rclcpp::ok()) rclcpp::shutdown();
    return 2;
  }
}
