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
#include <moveit/collision_detection_bullet/collision_detector_allocator_bullet.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/mesh.hpp>

namespace fs = std::filesystem;
struct Candidate { int waypoint; std::string id; std::vector<double> q; };

static std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out; std::string field; bool quoted = false;
  for (char c : s) { if (c == '"') quoted = !quoted; else if (c == ',' && !quoted) { out.push_back(field); field.clear(); } else field += c; }
  out.push_back(field); return out;
}
static double number(const std::string& s) { try { return std::stod(s); } catch (...) { return 0.0; } }
static std::vector<double> vector_field(std::string s) {
  auto a = s.find('['), b = s.rfind(']'); if (a != std::string::npos) s = s.substr(a + 1, b > a ? b - a - 1 : s.size());
  std::vector<double> out; for (const auto& x : split_csv(s)) out.push_back(number(x)); return out;
}
static std::vector<Candidate> read_candidates(const fs::path& path) {
  std::ifstream f(path); std::string line; std::getline(f, line); auto header = split_csv(line); std::map<std::string, std::size_t> col; for (std::size_t i = 0; i < header.size(); ++i) col[header[i]] = i;
  std::vector<Candidate> out; while (std::getline(f, line)) { if (line.empty()) continue; auto row = split_csv(line); auto q = vector_field(row[col["joint_values"]]); if (q.size() == 6) out.push_back({static_cast<int>(number(row[col["waypoint_id"]])), row[col["candidate_id"]], q}); } return out;
}

struct MeshBuilder {
  shape_msgs::msg::Mesh mesh; std::map<std::string, std::uint32_t> indices;
  std::uint32_t add(double x, double y, double z) { std::ostringstream key; key << std::setprecision(17) << x << "," << y << "," << z; auto it = indices.find(key.str()); if (it != indices.end()) return it->second; geometry_msgs::msg::Point p; p.x = x; p.y = y; p.z = z; auto id = static_cast<std::uint32_t>(mesh.vertices.size()); mesh.vertices.push_back(p); indices[key.str()] = id; return id; }
};
static shape_msgs::msg::Mesh read_mesh(const fs::path& path) {
  std::ifstream f(path); if (!f) throw std::runtime_error("cannot read collision part: " + path.string());
  MeshBuilder b; std::vector<std::array<double, 3>> facet; std::string line;
  while (std::getline(f, line)) { std::stringstream ss(line); std::string word; ss >> word; if (word != "vertex") continue; std::array<double, 3> v{}; ss >> v[0] >> v[1] >> v[2]; facet.push_back(v); if (facet.size() == 3) { shape_msgs::msg::MeshTriangle t; for (int i = 0; i < 3; ++i) t.vertex_indices[i] = b.add(facet[i][0], facet[i][1], facet[i][2]); b.mesh.triangles.push_back(t); facet.clear(); } }
  return b.mesh;
}
static std::vector<shape_msgs::msg::Mesh> read_parts(const fs::path& dir, std::vector<std::string>& names) {
  for (const auto& e : fs::directory_iterator(dir)) if (e.path().extension() == ".stl") names.push_back(e.path().filename().string()); std::sort(names.begin(), names.end()); std::vector<shape_msgs::msg::Mesh> out; for (const auto& name : names) out.push_back(read_mesh(dir / name)); return out;
}
static void json_string(std::ostream& out, const std::string& value) { out << '"'; for (char c : value) { if (c == '"') out << "\\\""; else if (c == '\\') out << "\\\\"; else out << c; } out << '"'; }
static std::string json_bool(bool value) { return value ? "true" : "false"; }
static std::string json_number(double value) { std::ostringstream s; s << std::setprecision(17) << value; return s.str(); }

struct CollisionSummary { bool collision = false, self = false, world = false; std::vector<std::string> pairs; std::vector<double> depths; std::vector<Eigen::Vector3d> positions; };
static CollisionSummary check(planning_scene::PlanningScene& scene, moveit::core::RobotState& state, const std::string& group) {
  collision_detection::CollisionRequest request; request.group_name = group; request.contacts = true; request.max_contacts = 4096; request.max_contacts_per_pair = 64; request.verbose = true; request.pad_environment_collisions = true;
  collision_detection::CollisionResult result; result.clear(); scene.checkCollision(request, result, state); CollisionSummary out; out.collision = result.collision;
  for (const auto& item : result.contacts) for (const auto& contact : item.second) { out.pairs.push_back(contact.body_name_1 + "<->" + contact.body_name_2); bool a = contact.body_type_1 != collision_detection::BodyTypes::WORLD_OBJECT, b = contact.body_type_2 != collision_detection::BodyTypes::WORLD_OBJECT; out.self = out.self || (a && b); out.world = out.world || (a != b); out.depths.push_back(contact.depth); out.positions.push_back(contact.pos); }
  std::sort(out.pairs.begin(), out.pairs.end()); out.pairs.erase(std::unique(out.pairs.begin(), out.pairs.end()), out.pairs.end()); return out;
}
static moveit_msgs::msg::CollisionObject make_object(const fs::path& dir, const std::vector<std::string>& names, const std::vector<shape_msgs::msg::Mesh>& meshes) {
  moveit_msgs::msg::CollisionObject object; object.header.frame_id = "base_link"; object.id = "horseshoe_collision_compound"; object.meshes = meshes; object.mesh_poses.resize(meshes.size()); for (auto& pose : object.mesh_poses) { pose.orientation.w = 1.0; pose.position.x = 0.32; pose.position.y = -0.05; pose.position.z = -0.38; } object.operation = moveit_msgs::msg::CollisionObject::ADD; return object;
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv); auto node = std::make_shared<rclcpp::Node>("stage23a5_native_probe", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  std::string candidate_csv, parts_dir, output_dir, backend, group; int run_index = 1; node->get_parameter_or("candidate_csv", candidate_csv, std::string{}); node->get_parameter_or("parts_dir", parts_dir, std::string{}); node->get_parameter_or("output_dir", output_dir, std::string{}); node->get_parameter_or("backend", backend, std::string("fcl")); node->get_parameter_or("run_index", run_index, 1); node->get_parameter_or("group_name", group, std::string("fairino5_v6_group"));
  if (candidate_csv.empty() || parts_dir.empty() || output_dir.empty()) return 2;
  fs::create_directories(output_dir); std::vector<std::string> names; auto meshes = read_parts(parts_dir, names); auto candidates = read_candidates(candidate_csv);
  robot_model_loader::RobotModelLoader::Options options("robot_description"); options.load_kinematics_solvers = false; robot_model_loader::RobotModelLoader loader(node, options); auto model = loader.getModel(); if (!model) return 3; auto scene = std::make_unique<planning_scene::PlanningScene>(model); scene->processCollisionObjectMsg(make_object(parts_dir, names, meshes)); if (backend == "bullet") scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create()); else scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create()); auto* joint_group = model->getJointModelGroup(group); if (!joint_group) return 4;
  const fs::path jsonl = fs::path(output_dir) / (backend + "_candidate_results_run" + std::to_string(run_index) + ".jsonl"); std::ofstream out(jsonl); int valid = 0, colliding = 0; std::map<int, bool> waypoint_valid; moveit::core::RobotState state(model);
  for (const auto& candidate : candidates) { state.setJointGroupPositions(joint_group, candidate.q); state.update(); auto result = check(*scene, state, group); if (result.collision) ++colliding; else { ++valid; waypoint_valid[candidate.waypoint] = true; }
    out << "{\"candidate_id\":"; json_string(out, candidate.id); out << ",\"waypoint_id\":" << candidate.waypoint << ",\"joint_values\":["; for (std::size_t i = 0; i < candidate.q.size(); ++i) { if (i) out << ','; out << json_number(candidate.q[i]); } out << "],\"backend\":"; json_string(out, backend); out << ",\"collision\":" << json_bool(result.collision) << ",\"valid\":" << json_bool(!result.collision) << ",\"self_collision\":" << json_bool(result.self) << ",\"robot_world_collision\":" << json_bool(result.world) << ",\"collision_object_pairs\":["; for (std::size_t i = 0; i < result.pairs.size(); ++i) { if (i) out << ','; json_string(out, result.pairs[i]); } out << "],\"penetration_depths\":["; for (std::size_t i = 0; i < result.depths.size(); ++i) { if (i) out << ','; out << json_number(result.depths[i]); } out << "],\"contact_positions\":["; for (std::size_t i = 0; i < result.positions.size(); ++i) { if (i) out << ','; out << '[' << json_number(result.positions[i].x()) << ',' << json_number(result.positions[i].y()) << ',' << json_number(result.positions[i].z()) << ']'; } out << "]}\n";
  }
  out.close(); std::ofstream summary(fs::path(output_dir) / (backend + "_run" + std::to_string(run_index) + "_summary.json")); summary << "{\n  \"backend\": "; json_string(summary, backend); summary << ",\n  \"run_index\": " << run_index << ",\n  \"candidate_count\": " << candidates.size() << ",\n  \"valid_node_count\": " << valid << ",\n  \"colliding_candidate_count\": " << colliding << ",\n  \"valid_waypoint_count\": " << waypoint_valid.size() << ",\n  \"collision_object_id\": \"horseshoe_collision_compound\",\n  \"collision_part_count\": " << meshes.size() << ",\n  \"mesh_scale\": [1.0, 1.0, 1.0],\n  \"padding\": 0.0,\n  \"acm_modified\": false,\n  \"scene_built_from_scratch\": true,\n  \"collision_method\": \"adaptive_discrete_interpolation\",\n  \"ccd_status\": \"not_available\",\n  \"clearance_status\": \"not_available\"\n}\n";
  if (run_index == 1 && backend == "fcl") { std::ofstream provenance(fs::path(output_dir) / "runtime_provenance.json"); provenance << "{\n  \"status\": \"native_runtime_observed\",\n  \"ros2_distribution\": \"jazzy\",\n  \"moveit_version\": \"2.12.4\",\n  \"backend_probe\": \"moveit_planning_scene\",\n  \"fcl_backend\": \"collision_detection_fcl\",\n  \"bullet_backend\": \"collision_detection_bullet\",\n  \"collision_method\": \"adaptive_discrete_interpolation\",\n  \"ccd_status\": \"not_available\",\n  \"clearance_status\": \"not_available\",\n  \"acm_modified\": false,\n  \"padding\": 0.0,\n  \"scale\": [1.0, 1.0, 1.0],\n  \"robot_description_source\": \"ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro\",\n  \"semantic_source\": \"ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf\",\n  \"moveit_library_path\": \"/opt/ros/jazzy/lib\",\n  \"fcl_library_path\": \"/lib/x86_64-linux-gnu/libfcl.so.0.7\",\n  \"bullet_library_path\": \"/lib/x86_64-linux-gnu/libBulletCollision.so.3.24\"\n}\n"; }
  rclcpp::shutdown(); return 0;
}
