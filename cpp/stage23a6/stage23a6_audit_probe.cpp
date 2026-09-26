#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/point.hpp>
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection/collision_matrix.hpp>
#include <moveit/collision_detection_bullet/collision_detector_allocator_bullet.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit_msgs/msg/allowed_collision_matrix.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/mesh.hpp>

namespace fs = std::filesystem;

struct Candidate { int waypoint = -1; std::string id; std::vector<double> q; };

static std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out; std::string field; bool quoted = false;
  for (char c : s) {
    if (c == '"') quoted = !quoted;
    else if (c == ',' && !quoted) { out.push_back(field); field.clear(); }
    else field += c;
  }
  out.push_back(field); return out;
}
static double number(const std::string& s) { try { return std::stod(s); } catch (...) { return 0.0; } }
static std::vector<double> vector_field(std::string s) {
  auto a = s.find('['), b = s.rfind(']');
  if (a != std::string::npos) s = s.substr(a + 1, b > a ? b - a - 1 : s.size());
  std::vector<double> out; for (const auto& x : split_csv(s)) out.push_back(number(x)); return out;
}
static std::vector<Candidate> read_candidates(const fs::path& path) {
  std::ifstream f(path); std::string line; if (!std::getline(f, line)) throw std::runtime_error("empty candidate CSV");
  auto header = split_csv(line); std::map<std::string, std::size_t> col;
  for (std::size_t i = 0; i < header.size(); ++i) col[header[i]] = i;
  std::vector<Candidate> out;
  while (std::getline(f, line)) {
    if (line.empty()) continue; auto row = split_csv(line); auto q = vector_field(row[col["joint_values"]]);
    if (q.size() == 6) out.push_back({static_cast<int>(number(row[col["waypoint_id"]])), row[col["candidate_id"]], q});
  }
  return out;
}

struct MeshBuilder {
  shape_msgs::msg::Mesh mesh; std::map<std::string, std::uint32_t> indices;
  std::uint32_t add(double x, double y, double z) {
    std::ostringstream key; key << std::setprecision(17) << x << "," << y << "," << z;
    auto it = indices.find(key.str()); if (it != indices.end()) return it->second;
    geometry_msgs::msg::Point p; p.x = x; p.y = y; p.z = z;
    auto id = static_cast<std::uint32_t>(mesh.vertices.size()); mesh.vertices.push_back(p); indices[key.str()] = id; return id;
  }
};
static shape_msgs::msg::Mesh read_mesh(const fs::path& path) {
  std::ifstream f(path); if (!f) throw std::runtime_error("cannot read collision part: " + path.string());
  MeshBuilder b; std::vector<std::array<double, 3>> facet; std::string line;
  while (std::getline(f, line)) {
    std::stringstream ss(line); std::string word; ss >> word; if (word != "vertex") continue;
    std::array<double, 3> v{}; ss >> v[0] >> v[1] >> v[2]; facet.push_back(v);
    if (facet.size() == 3) { shape_msgs::msg::MeshTriangle t; for (int i = 0; i < 3; ++i) t.vertex_indices[i] = b.add(facet[i][0], facet[i][1], facet[i][2]); b.mesh.triangles.push_back(t); facet.clear(); }
  }
  return b.mesh;
}
static std::vector<shape_msgs::msg::Mesh> read_parts(const fs::path& dir, std::vector<std::string>& names) {
  for (const auto& e : fs::directory_iterator(dir)) if (e.path().extension() == ".stl") names.push_back(e.path().filename().string());
  std::sort(names.begin(), names.end()); std::vector<shape_msgs::msg::Mesh> out;
  for (const auto& name : names) out.push_back(read_mesh(dir / name)); return out;
}
static void json_string(std::ostream& out, const std::string& value) {
  out << '"'; for (char c : value) { if (c == '"') out << "\\\""; else if (c == '\\') out << "\\\\"; else if (c == '\n') out << "\\n"; else out << c; } out << '"';
}
static std::string json_bool(bool value) { return value ? "true" : "false"; }
static std::string json_number(double value) { std::ostringstream s; s << std::setprecision(17) << value; return s.str(); }
static std::string body_type(collision_detection::BodyType t) {
  if (t == collision_detection::BodyTypes::ROBOT_LINK) return "ROBOT_LINK";
  if (t == collision_detection::BodyTypes::ROBOT_ATTACHED) return "ROBOT_ATTACHED";
  return "WORLD_OBJECT";
}
static std::string pair_key(std::string a, std::string b) { if (a > b) std::swap(a, b); return a + "<->" + b; }
static void vec3(std::ostream& out, const Eigen::Vector3d& v) { out << '[' << json_number(v.x()) << ',' << json_number(v.y()) << ',' << json_number(v.z()) << ']'; }
static void transform_json(std::ostream& out, const Eigen::Isometry3d& t) {
  out << "{\"matrix\":[";
  for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) { if (r || c) out << ','; out << json_number(t(r, c)); }
  out << "],\"translation\":"; vec3(out, t.translation()); out << '}';
}

struct Check {
  bool collision = false;
  std::vector<std::string> pairs;
  std::vector<collision_detection::Contact> contacts;
};
static Check collect(const collision_detection::CollisionResult& result) {
  Check out; out.collision = result.collision;
  for (const auto& item : result.contacts) {
    for (const auto& c : item.second) { out.pairs.push_back(pair_key(c.body_name_1, c.body_name_2)); out.contacts.push_back(c); }
  }
  std::sort(out.pairs.begin(), out.pairs.end()); out.pairs.erase(std::unique(out.pairs.begin(), out.pairs.end()), out.pairs.end());
  return out;
}
static void write_contacts(std::ostream& out, const std::vector<collision_detection::Contact>& contacts) {
  out << '[';
  for (std::size_t i = 0; i < contacts.size(); ++i) {
    if (i) out << ','; const auto& c = contacts[i]; out << "{\"link_1\":"; json_string(out, c.body_name_1);
    out << ",\"link_2\":"; json_string(out, c.body_name_2); out << ",\"sorted_link_pair\":"; json_string(out, pair_key(c.body_name_1, c.body_name_2));
    out << ",\"body_type_1\":"; json_string(out, body_type(c.body_type_1)); out << ",\"body_type_2\":"; json_string(out, body_type(c.body_type_2));
    out << ",\"shape_index_1\":\"unavailable_moveit_contact_api\",\"shape_index_2\":\"unavailable_moveit_contact_api\"";
    out << ",\"shape_type_1\":\"unavailable_moveit_contact_api\",\"shape_type_2\":\"unavailable_moveit_contact_api\"";
    out << ",\"contact_position\":"; vec3(out, c.pos); out << ",\"contact_normal\":"; vec3(out, c.normal);
    out << ",\"penetration_depth\":" << json_number(c.depth) << ",\"distance\":\"unavailable_collision_contact_has_no_distance_field\"";
    out << ",\"nearest_points\":["; vec3(out, c.nearest_points[0]); out << ','; vec3(out, c.nearest_points[1]); out << "]}";
  }
  out << ']';
}
static moveit_msgs::msg::CollisionObject make_object(const std::vector<shape_msgs::msg::Mesh>& meshes) {
  moveit_msgs::msg::CollisionObject object; object.header.frame_id = "base_link"; object.id = "horseshoe_collision_compound";
  object.meshes = meshes; object.mesh_poses.resize(meshes.size());
  for (auto& pose : object.mesh_poses) { pose.orientation.w = 1.0; pose.position.x = 0.32; pose.position.y = -0.05; pose.position.z = -0.38; }
  object.operation = moveit_msgs::msg::CollisionObject::ADD; return object;
}
static void write_acm(const fs::path& path, const collision_detection::AllowedCollisionMatrix& acm) {
  moveit_msgs::msg::AllowedCollisionMatrix msg; acm.getMessage(msg); std::ofstream out(path);
  out << "{\n\"entry_names\":["; for (std::size_t i = 0; i < msg.entry_names.size(); ++i) { if (i) out << ','; json_string(out, msg.entry_names[i]); }
  out << "],\n\"entry_values\":["; for (std::size_t i = 0; i < msg.entry_values.size(); ++i) { if (i) out << ','; out << '['; for (std::size_t j = 0; j < msg.entry_values[i].enabled.size(); ++j) { if (j) out << ','; out << json_bool(msg.entry_values[i].enabled[j]); } out << ']'; }
  out << "],\n\"default_entry_names\":["; for (std::size_t i = 0; i < msg.default_entry_names.size(); ++i) { if (i) out << ','; json_string(out, msg.default_entry_names[i]); }
  out << "],\n\"default_entry_values\":["; for (std::size_t i = 0; i < msg.default_entry_values.size(); ++i) { if (i) out << ','; out << json_bool(msg.default_entry_values[i]); } out << "],\n\"size\":" << acm.getSize() << "\n}\n";
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv); auto node = std::make_shared<rclcpp::Node>("stage23a6_audit_probe", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  std::string candidate_csv, parts_dir, output_dir, backend, group; int run_index = 1;
  node->get_parameter_or("candidate_csv", candidate_csv, std::string{}); node->get_parameter_or("parts_dir", parts_dir, std::string{}); node->get_parameter_or("output_dir", output_dir, std::string{});
  node->get_parameter_or("backend", backend, std::string("fcl")); node->get_parameter_or("run_index", run_index, 1); node->get_parameter_or("group_name", group, std::string("fairino5_v6_group"));
  if (candidate_csv.empty() || parts_dir.empty() || output_dir.empty()) return 2; fs::create_directories(output_dir);
  std::vector<std::string> names; auto meshes = read_parts(parts_dir, names); auto candidates = read_candidates(candidate_csv);
  robot_model_loader::RobotModelLoader::Options options("robot_description"); options.load_kinematics_solvers = false; robot_model_loader::RobotModelLoader loader(node, options); auto model = loader.getModel(); if (!model) return 3;
  auto scene = std::make_unique<planning_scene::PlanningScene>(model); scene->processCollisionObjectMsg(make_object(meshes));
  if (backend == "bullet") scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create()); else scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  auto* joint_group = model->getJointModelGroup(group); if (!joint_group) return 4; write_acm(fs::path(output_dir) / (backend + "_acm_run" + std::to_string(run_index) + ".json"), scene->getAllowedCollisionMatrix());
  const fs::path jsonl = fs::path(output_dir) / (backend + "_classification_run" + std::to_string(run_index) + ".jsonl"); std::ofstream out(jsonl); moveit::core::RobotState state(model);
  collision_detection::CollisionRequest request; request.group_name = group; request.contacts = true; request.max_contacts = 4096; request.max_contacts_per_pair = 256; request.verbose = false; request.pad_environment_collisions = true; request.pad_self_collisions = false;
  for (const auto& candidate : candidates) {
    state.setJointGroupPositions(joint_group, candidate.q); state.update(); collision_detection::CollisionResult self_result, world_result, combined_result;
    scene->checkSelfCollision(request, self_result, state); scene->getCollisionEnv()->checkRobotCollision(request, world_result, state); scene->checkCollision(request, combined_result, state);
    auto self = collect(self_result), world = collect(world_result), combined = collect(combined_result);
    collision_detection::DistanceRequest dreq; dreq.group_name = group; dreq.type = collision_detection::DistanceRequestTypes::GLOBAL; dreq.enable_nearest_points = true; dreq.enable_signed_distance = true; collision_detection::DistanceResult dres; scene->getCollisionEnv()->distanceSelf(dreq, dres, state);
    out << "{\"candidate_id\":"; json_string(out, candidate.id); out << ",\"waypoint_id\":" << candidate.waypoint << ",\"joint_values\":["; for (std::size_t i = 0; i < candidate.q.size(); ++i) { if (i) out << ','; out << json_number(candidate.q[i]); } out << "]";
    out << ",\"backend\":"; json_string(out, backend); out << ",\"self_collision\":" << json_bool(self.collision) << ",\"robot_world_collision\":" << json_bool(world.collision) << ",\"combined_collision\":" << json_bool(combined.collision);
    auto write_pairs = [&](const char* name, const std::vector<std::string>& pairs) { out << ",\"" << name << "\":["; for (std::size_t i = 0; i < pairs.size(); ++i) { if (i) out << ','; json_string(out, pairs[i]); } out << ']'; };
    write_pairs("self_collision_pairs", self.pairs); write_pairs("robot_world_collision_pairs", world.pairs); write_pairs("combined_collision_pairs", combined.pairs);
    out << ",\"self_contacts\":"; write_contacts(out, self.contacts); out << ",\"robot_world_contacts\":"; write_contacts(out, world.contacts); out << ",\"combined_contacts\":"; write_contacts(out, combined.contacts);
    std::map<std::string, Eigen::Isometry3d> transforms;
    for (const auto& c : self.contacts) {
      if (auto* l1 = model->getLinkModel(c.body_name_1)) transforms[c.body_name_1] = state.getGlobalLinkTransform(l1);
      if (auto* l2 = model->getLinkModel(c.body_name_2)) transforms[c.body_name_2] = state.getGlobalLinkTransform(l2);
    }
    out << ",\"self_contact_link_world_transforms\":{";
    bool first_transform = true;
    for (const auto& item : transforms) { if (!first_transform) out << ','; first_transform = false; json_string(out, item.first); out << ':'; transform_json(out, item.second); }
    out << '}';
    out << ",\"distance_self\":" << json_number(dres.minimum_distance.distance) << ",\"distance_self_link_1\":"; json_string(out, dres.minimum_distance.link_names[0]); out << ",\"distance_self_link_2\":"; json_string(out, dres.minimum_distance.link_names[1]);
    out << ",\"distance_self_nearest_points\":["; vec3(out, dres.minimum_distance.nearest_points[0]); out << ','; vec3(out, dres.minimum_distance.nearest_points[1]); out << "]}" << '\n';
  }
  std::ofstream summary(fs::path(output_dir) / (backend + "_classification_run" + std::to_string(run_index) + "_summary.json")); summary << "{\n\"backend\":"; json_string(summary, backend); summary << ",\"run_index\":" << run_index << ",\"candidate_count\":" << candidates.size() << ",\"collision_method\":\"adaptive_discrete_interpolation\",\"ccd_status\":\"not_available\",\"clearance_status\":\"not_available\",\"shape_indices_status\":\"unavailable_moveit_contact_api\"\n}\n";
  rclcpp::shutdown(); return 0;
}
