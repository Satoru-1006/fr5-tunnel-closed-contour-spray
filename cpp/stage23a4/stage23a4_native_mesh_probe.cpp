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
struct PoseRow { double x{}, y{}, z{}, nx{}, ny{}, nz{}; };
struct CollisionSummary { bool full=false, self=false, world=false; std::string pairs; std::size_t contacts=0; };

static std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out; std::string field; bool quoted=false;
  for (char c : s) { if (c=='"') quoted=!quoted; else if (c==',' && !quoted) { out.push_back(field); field.clear(); } else field+=c; }
  out.push_back(field); return out;
}
static double number(const std::string& s) { try { return std::stod(s); } catch (...) { return 0.0; } }
static std::vector<double> vector_field(std::string s) {
  const auto a=s.find('['), b=s.rfind(']'); if (a!=std::string::npos) s=s.substr(a+1, b>a ? b-a-1 : s.size());
  std::vector<double> out; for (const auto& x : split(s)) out.push_back(number(x)); return out;
}
static std::vector<Candidate> read_candidates(const fs::path& path) {
  std::ifstream f(path); std::string line; std::getline(f,line); auto header=split(line); std::map<std::string,std::size_t> col;
  for (std::size_t i=0;i<header.size();++i) col[header[i]]=i; std::vector<Candidate> out;
  while (std::getline(f,line)) { if (line.empty()) continue; auto row=split(line); auto q=vector_field(row[col["joint_values"]]); if(q.size()==6) out.push_back({int(number(row[col["waypoint_id"]])),row[col["candidate_id"]],q}); }
  return out;
}
static std::vector<PoseRow> read_poses(const fs::path& path) {
  std::ifstream f(path); std::string line; std::getline(f,line); auto header=split(line); std::map<std::string,std::size_t> col;
  for (std::size_t i=0;i<header.size();++i) col[header[i]]=i; std::vector<PoseRow> out;
  while(std::getline(f,line)){if(line.empty())continue;auto row=split(line);auto get=[&](const char* n){return number(row[col[n]]);};out.push_back({get("x"),get("y"),get("z"),get("nx"),get("ny"),get("nz")});}
  return out;
}

struct MeshBuilder {
  shape_msgs::msg::Mesh mesh;
  std::map<std::string,std::uint32_t> indices;
  std::uint32_t add(const Eigen::Vector3d& v) {
    std::ostringstream key; key<<std::setprecision(17)<<v.x()<<","<<v.y()<<","<<v.z();
    auto it=indices.find(key.str()); if(it!=indices.end()) return it->second;
    geometry_msgs::msg::Point p; p.x=v.x(); p.y=v.y(); p.z=v.z(); auto id=static_cast<std::uint32_t>(mesh.vertices.size()); mesh.vertices.push_back(p); indices[key.str()]=id; return id;
  }
};
static int component_for(const Eigen::Vector3d& c) {
  if (c.z() < 0.16) return 0;              // bottom wall and its sealed rim
  if (c.y() > 0.38) return 1;              // +y wall and rim
  if (c.y() < -0.38) return 2;             // -y wall and rim
  return 3;                                // crown and rim
}
static std::vector<shape_msgs::msg::Mesh> read_mesh_components(const fs::path& path) {
  std::ifstream f(path); if(!f) throw std::runtime_error("cannot read ASCII STL");
  std::string line; std::vector<Eigen::Vector3d> facet; std::vector<MeshBuilder> builders(4);
  while(std::getline(f,line)){std::stringstream ss(line);std::string word;ss>>word;if(word!="vertex")continue;double x,y,z;ss>>x>>y>>z;facet.emplace_back(x,y,z);if(facet.size()!=3)continue;Eigen::Vector3d c=(facet[0]+facet[1]+facet[2])/3.0;auto& b=builders[component_for(c)];shape_msgs::msg::MeshTriangle tri;tri.vertex_indices={b.add(facet[0]),b.add(facet[1]),b.add(facet[2])};b.mesh.triangles.push_back(tri);facet.clear();}
  std::vector<shape_msgs::msg::Mesh> out;for(auto&b:builders)if(!b.mesh.triangles.empty())out.push_back(std::move(b.mesh));return out;
}
static moveit_msgs::msg::CollisionObject make_mesh_object(const fs::path& path, const std::vector<double>& xyz) {
  auto meshes=read_mesh_components(path); moveit_msgs::msg::CollisionObject o; o.header.frame_id="base_link"; o.id="scaled_demo_tunnel"; o.meshes=std::move(meshes); geometry_msgs::msg::Pose pose; pose.position.x=xyz.size()>0?xyz[0]:.32; pose.position.y=xyz.size()>1?xyz[1]:-.05; pose.position.z=xyz.size()>2?xyz[2]:-.38; pose.orientation.w=1.0; o.mesh_poses.push_back(pose); o.operation=moveit_msgs::msg::CollisionObject::ADD; return o;
}
static CollisionSummary check(planning_scene::PlanningScene& scene, moveit::core::RobotState& state, const std::string& group) {
  collision_detection::CollisionRequest request; request.group_name=group; request.contacts=true; request.max_contacts=4096; request.max_contacts_per_pair=64; request.verbose=true; request.pad_environment_collisions=true;
  collision_detection::CollisionResult result; result.clear(); scene.checkCollision(request,result,state); CollisionSummary out; out.full=result.collision; out.contacts=result.contact_count; std::vector<std::string> pairs;
  for(const auto& item:result.contacts) for(const auto& contact:item.second){pairs.push_back(contact.body_name_1+"<->"+contact.body_name_2);bool a=contact.body_type_1!=collision_detection::BodyTypes::WORLD_OBJECT,b=contact.body_type_2!=collision_detection::BodyTypes::WORLD_OBJECT;out.self=out.self||(a&&b);out.world=out.world||((a&&!b)||(!a&&b));}
  std::sort(pairs.begin(),pairs.end());pairs.erase(std::unique(pairs.begin(),pairs.end()),pairs.end());for(std::size_t i=0;i<pairs.size();++i){if(i)out.pairs+='|';out.pairs+=pairs[i];}return out;
}
int main(int argc,char** argv) {
  rclcpp::init(argc,argv); auto node=std::make_shared<rclcpp::Node>("stage23a4_native_mesh_probe",rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  std::string candidate_path,pose_path,mesh_path,output_path,backend,group;int run_index=1;node->get_parameter_or("candidate_csv",candidate_path,std::string{});node->get_parameter_or("pose_csv",pose_path,std::string{});node->get_parameter_or("mesh_path",mesh_path,std::string{});node->get_parameter_or("output_dir",output_path,std::string{});node->get_parameter_or("backend",backend,std::string("fcl"));node->get_parameter_or("run_index",run_index,1);node->get_parameter_or("group_name",group,std::string("fairino5_v6_group"));if(candidate_path.empty()||pose_path.empty()||mesh_path.empty()||output_path.empty())return 2;
  std::vector<double> mesh_xyz; node->get_parameter_or("mesh_pose_xyz",mesh_xyz,std::vector<double>{.32,-.05,-.38}); fs::create_directories(output_path);robot_model_loader::RobotModelLoader::Options options("robot_description");options.load_kinematics_solvers=false;robot_model_loader::RobotModelLoader loader(node,options);auto model=loader.getModel();if(!model)return 3;auto candidates=read_candidates(candidate_path);auto poses=read_poses(pose_path);(void)poses;auto scene=std::make_unique<planning_scene::PlanningScene>(model);scene->processCollisionObjectMsg(make_mesh_object(mesh_path,mesh_xyz));if(backend=="bullet")scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create());else scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());auto* joint_group=model->getJointModelGroup(group);if(!joint_group)return 4;
  std::ofstream out(fs::path(output_path)/"candidate_results.csv");out<<"candidate_id,waypoint_id,self_collision,robot_world_collision,validity,collision_pairs,contact_count,backend,run_index\n";int valid=0,colliding=0;moveit::core::RobotState state(model);for(const auto& candidate:candidates){state.setJointGroupPositions(joint_group,candidate.q);state.update();auto result=check(*scene,state,group);if(result.full)++colliding;else ++valid;out<<candidate.id<<","<<candidate.waypoint<<","<<(result.self?"true":"false")<<","<<(result.world?"true":"false")<<","<<(!result.full?"true":"false")<<","<<result.pairs<<","<<result.contacts<<","<<backend<<","<<run_index<<"\n";}out.close();std::map<int,bool> waypoint_valid;std::ifstream rows(fs::path(output_path)/"candidate_results.csv");std::string line;std::getline(rows,line);while(std::getline(rows,line)){auto fields=split(line);if(fields.size()>4&&fields[4]=="true")waypoint_valid[int(number(fields[1]))]=true;}int valid_waypoints=0;for(const auto& item:waypoint_valid)valid_waypoints+=item.second?1:0;
  std::ofstream summary(fs::path(output_path)/"run_summary.json");summary<<"{\n  \"backend\": \""<<backend<<"\",\n  \"run_index\": "<<run_index<<",\n  \"candidate_count\": "<<candidates.size()<<",\n  \"valid_node_count\": "<<valid<<",\n  \"colliding_candidate_count\": "<<colliding<<",\n  \"valid_waypoint_count\": "<<valid_waypoints<<",\n  \"collision_object_count\": 1,\n  \"planning_scene_mesh_component_count\": 4,\n  \"mesh_scale\": [1.0,1.0,1.0],\n  \"scene_built_from_scratch\": true,\n  \"collision_method\": \"adaptive_discrete_interpolation\",\n  \"ccd_status\": \"not_available\",\n  \"clearance_status\": \"not_available\"\n}\n";rclcpp::shutdown();return 0;
}
