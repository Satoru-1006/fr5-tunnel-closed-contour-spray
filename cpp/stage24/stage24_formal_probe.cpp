// Native Stage 2.4 edge checker.  The scene helpers are the exact helpers
// used by the frozen Stage 2.3B formal probe; only the edge loop is new.
#define main stage23a7_formal_helpers_original_main
#include "../stage23a7/stage23a7_runtime_audit_probe.cpp"
#undef main

#include <cstdlib>
#include <cmath>
#include <limits>

struct EdgeRequest {
  int from_waypoint = -1;
  int to_waypoint = -1;
  std::string from_id;
  std::string to_id;
  std::vector<double> q0;
  std::vector<double> q1;
  double max_step_deg = 0.0;
};

static bool stage24_has_self_pair(const Check& check)
{
  for (const auto& pair : check.pairs)
    if (pair.find("horseshoe_collision_compound") == std::string::npos)
      return true;
  return false;
}

static bool stage24_has_world_pair(const Check& check)
{
  for (const auto& pair : check.pairs)
    if (pair.find("horseshoe_collision_compound") != std::string::npos)
      return true;
  return false;
}

static std::vector<EdgeRequest> read_edges(const fs::path& path)
{
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("edge request open failed: " + path.string());
  std::string line;
  if (!std::getline(in, line))
    throw std::runtime_error("empty edge request file");
  const auto header = split_csv(line);
  std::map<std::string, size_t> col;
  for (size_t i = 0; i < header.size(); ++i) {
    auto name = header[i];
    if (!name.empty() && name.back() == '\r')
      name.pop_back();
    col[name] = i;
  }
  const auto require = [&](const std::string& name) -> size_t {
    auto it = col.find(name);
    if (it == col.end())
      throw std::runtime_error("missing edge column: " + name);
    return it->second;
  };
  const size_t from_wp = require("from_waypoint");
  const size_t to_wp = require("to_waypoint");
  const size_t from_id = require("from_candidate_id");
  const size_t to_id = require("to_candidate_id");
  const size_t interpolation_step = require("interpolation_step_deg");
  std::vector<size_t> q0(6), q1(6);
  for (int i = 0; i < 6; ++i) {
    q0[i] = require("q0_" + std::to_string(i));
    q1[i] = require("q1_" + std::to_string(i));
  }
  std::vector<EdgeRequest> edges;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    const auto row = split_csv(line);
    EdgeRequest e;
    e.from_waypoint = static_cast<int>(num(row[from_wp]));
    e.to_waypoint = static_cast<int>(num(row[to_wp]));
    e.from_id = row[from_id];
    e.to_id = row[to_id];
    e.max_step_deg = num(row[interpolation_step]);
    e.q0.resize(6);
    e.q1.resize(6);
    for (int i = 0; i < 6; ++i) {
      e.q0[i] = num(row[q0[i]]);
      e.q1[i] = num(row[q1[i]]);
    }
    edges.push_back(std::move(e));
  }
  return edges;
}

static void write_string_array(std::ostream& out, const std::vector<std::string>& values)
{
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i)
      out << ',';
    js(out, values[i]);
  }
  out << ']';
}

static void write_first_contact(std::ostream& out, const Check& check)
{
  if (check.contacts.empty()) {
    out << "null";
    return;
  }
  write_contact_json(out, {check.contacts.front()});
}

static int run_probe(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
      "stage24_formal_probe", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  std::string edge_csv, parts_dir, output_dir, backend, group;
  int run = 1;
  node->get_parameter_or("edge_csv", edge_csv, std::string{});
  node->get_parameter_or("parts_dir", parts_dir, std::string{});
  node->get_parameter_or("output_dir", output_dir, std::string{});
  node->get_parameter_or("backend", backend, std::string("fcl"));
  node->get_parameter_or("run_index", run, 1);
  node->get_parameter_or("group_name", group, std::string("fairino5_v6_group"));
  if (edge_csv.empty() || parts_dir.empty() || output_dir.empty())
    return 2;

  // Stage 2.4C makes the runtime representation activation an explicit
  // formal precondition for both detectors.  A missing or invalid value must
  // fail closed instead of silently falling back to the historical Bullet
  // whole-hull representation.
  const char* shape_mode = std::getenv("FR5_BULLET_SHAPE_MODE");
  if (shape_mode == nullptr || std::string(shape_mode) != "use_shape_type")
  {
    std::cerr << "FR5_BULLET_SHAPE_MODE must equal use_shape_type for formal Stage 2.4 checks\n";
    rclcpp::shutdown();
    return 10;
  }

  fs::create_directories(output_dir);
  const auto edges = read_edges(edge_csv);
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
  if (!model->getJointModelGroup(group))
    return 4;
  if (backend == "bullet" && run == 1)
    dump_shapes(*scene, backend, output_dir, model);

  const auto request = req(group, true);
  std::ofstream out(fs::path(output_dir) / "native_edge_results.jsonl");
  std::ofstream samples;
  std::ofstream manifolds;
  if (run == 1)
  {
    samples.open(fs::path(output_dir) / "native_edge_samples.jsonl");
    if (backend == "bullet")
      manifolds.open(fs::path(output_dir) / "native_bullet_manifolds.jsonl");
  }
  std::ofstream provenance(fs::path(output_dir) / "runtime_provenance.json");
  provenance << "{\"status\":\"native_runtime_observed\",\"backend\":";
  js(provenance, backend);
  provenance << ",\"active_detector_name\":";
  js(provenance, scene->getCollisionDetectorName());
  provenance << ",\"collision_environment_concrete_class\":";
  js(provenance, demangle(typeid(*scene->getCollisionEnv()).name()));
  provenance << ",\"collision_method\":\"adaptive_discrete_interpolation\","
                "\"ccd_status\":\"not_available\",\"clearance_status\":\"not_available\","
                "\"formal_acm_modified\":false,\"request_contacts\":true,"
                "\"request_max_contacts\":524288,\"request_max_contacts_per_pair\":65536,"
                "\"pad_environment_collisions\":true,\"pad_self_collisions\":false,"
                "\"result_cleared_per_sample\":true,\"update_collision_body_transforms_per_sample\":true,"
                "\"sizeof_btScalar\":"
             << sizeof(btScalar) << "}\n";
  std::ifstream maps("/proc/self/maps");
  std::ofstream mapout(fs::path(output_dir) / "proc_self_maps.txt");
  mapout << maps.rdbuf();

  for (const auto& edge : edges) {
    const double max_step_rad = std::max(1.0e-12, edge.max_step_deg * M_PI / 180.0);
    int segment_steps = 1;
    for (size_t j = 0; j < edge.q0.size(); ++j)
      segment_steps = std::max(segment_steps, static_cast<int>(std::ceil(std::abs(edge.q1[j] - edge.q0[j]) / max_step_rad)));
    bool collision = false;
    int first_sample = -1;
    double first_t = std::numeric_limits<double>::quiet_NaN();
    Check first;
    int collision_sample_count = 0;
    for (int sample = 0; sample <= segment_steps; ++sample) {
      const double t = static_cast<double>(sample) / static_cast<double>(segment_steps);
      std::vector<double> q(edge.q0.size());
      for (size_t j = 0; j < q.size(); ++j)
        q[j] = edge.q0[j] + t * (edge.q1[j] - edge.q0[j]);
      RobotState state(model);
      state.setVariablePositions(q);
      const bool dirty_link_before = state.dirtyLinkTransforms();
      const bool dirty_collision_before = state.dirtyCollisionBodyTransforms();
      // Match the certified Stage 2.3B node path exactly.  This is the
      // documented prerequisite before PlanningScene collision checking.
      state.updateCollisionBodyTransforms();
      const bool dirty_link_after = state.dirtyLinkTransforms();
      const bool dirty_collision_after = state.dirtyCollisionBodyTransforms();
      collision_detection::CollisionResult result;
      result.clear();
      scene->checkCollision(request, result, state);
      const auto sample_check = collect(result);
      if (run == 1)
      {
        samples << "{\"from_waypoint\":" << edge.from_waypoint << ",\"to_waypoint\":" << edge.to_waypoint
                << ",\"from_candidate_id\":";
        js(samples, edge.from_id);
        samples << ",\"to_candidate_id\":";
        js(samples, edge.to_id);
        samples << ",\"backend\":";
        js(samples, backend);
        samples << ",\"sample_index\":" << sample << ",\"sample_count\":" << (segment_steps + 1)
                << ",\"t\":" << jn(t) << ",\"q\":[";
        for (size_t j = 0; j < q.size(); ++j)
        {
          if (j)
            samples << ',';
          samples << jn(q[j]);
        }
        samples << "],\"dirty_link_before\":" << (dirty_link_before ? "true" : "false")
                << ",\"dirty_collision_body_before\":" << (dirty_collision_before ? "true" : "false")
                << ",\"dirty_link_after\":" << (dirty_link_after ? "true" : "false")
                << ",\"dirty_collision_body_after\":" << (dirty_collision_after ? "true" : "false")
                << ",\"link_transforms\":[";
        const auto& links = model->getLinkModels();
        for (size_t link_index = 0; link_index < links.size(); ++link_index)
        {
          if (link_index)
            samples << ',';
          samples << "{\"link\":";
          js(samples, links[link_index]->getName());
          samples << ",\"transform\":";
          tf(samples, state.getGlobalLinkTransform(links[link_index]));
          samples << '}';
        }
        samples << "],\"collision\":" << (sample_check.collision ? "true" : "false")
                << ",\"collision_pairs\":";
        write_string_array(samples, sample_check.pairs);
        samples << ",\"self_collision\":" << (stage24_has_self_pair(sample_check) ? "true" : "false")
                << ",\"robot_world_collision\":" << (stage24_has_world_pair(sample_check) ? "true" : "false")
                << ",\"contacts\":";
        write_contact_json(samples, sample_check.contacts);
        samples << "}\n";
        if (backend == "bullet")
        {
          auto* bullet_env =
              dynamic_cast<collision_detection::CollisionEnvBullet*>(scene->getCollisionEnvNonConst().get());
          dump_manifolds(manifolds, "run1",
                         edge.from_id + "_to_" + edge.to_id + "_sample_" + std::to_string(sample),
                         edge.from_id + "_to_" + edge.to_id, std::to_string(edge.from_waypoint),
                         edge.from_id, bullet_env);
        }
      }
      if (result.collision) {
        ++collision_sample_count;
        if (!collision) {
          collision = true;
          first_sample = sample;
          first_t = t;
          first = collect(result);
        }
      }
    }
    out << "{\"from_waypoint\":" << edge.from_waypoint << ",\"to_waypoint\":" << edge.to_waypoint
        << ",\"from_candidate_id\":";
    js(out, edge.from_id);
    out << ",\"to_candidate_id\":";
    js(out, edge.to_id);
    out << ",\"backend\":";
    js(out, backend);
    out << ",\"collision_method\":\"adaptive_discrete_interpolation\",\"segment_steps\":"
        << segment_steps << ",\"samples_checked\":" << (segment_steps + 1)
        << ",\"collision_sample_count\":" << collision_sample_count << ",\"collision\":"
        << (collision ? "true" : "false") << ",\"status\":";
    js(out, collision ? "rejected" : "accepted");
    out << ",\"indeterminate\":false,\"first_collision_sample\":";
    if (first_sample < 0)
      out << "null";
    else
      out << first_sample;
    out << ",\"first_collision_t\":";
    if (first_sample < 0)
      out << "null";
    else
      out << jn(first_t);
    out << ",\"collision_pairs\":";
    write_string_array(out, first.pairs);
    out << ",\"self_collision\":" << (stage24_has_self_pair(first) ? "true" : "false")
        << ",\"robot_world_collision\":" << (stage24_has_world_pair(first) ? "true" : "false")
        << ",\"first_collision_contacts\":";
    write_first_contact(out, first);
    out << ",\"collision_result_cleared_per_sample\":true,\"state_update_called_per_sample\":true}\n";
  }
  provenance.close();
  rclcpp::shutdown();
  return 0;
}

int main(int argc, char** argv)
{
  try {
    return run_probe(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "stage24_formal_probe error: " << error.what() << '\n';
    if (rclcpp::ok())
      rclcpp::shutdown();
    return 10;
  }
}
