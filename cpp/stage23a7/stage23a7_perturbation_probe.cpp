#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#include <iomanip>
#include <limits>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/point.hpp>
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection_bullet/collision_detector_allocator_bullet.hpp>
#include <moveit/collision_detection_bullet/collision_env_bullet.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/mesh.hpp>
namespace fs=std::filesystem;
using collision_detection_bullet::CollisionObjectWrapper;

struct PerturbCallbackContext {
  std::ofstream* out = nullptr;
  std::string backend;
  std::string node_id;
  int waypoint_id = -1;
  std::string joint_name;
  double delta_rad = 0.0;
  std::size_t callback_index = 0;
};
static PerturbCallbackContext* g_perturb_callback_context = nullptr;
static std::string hex_float(float x) { std::uint32_t b=0; std::memcpy(&b,&x,sizeof(b)); std::ostringstream s; s<<"0x"<<std::hex<<std::setfill('0')<<std::setw(8)<<b; return s.str(); }
static void perturb_json_string(std::ostream& o,const std::string&s){o<<'"';for(char c:s){if(c=='"'||c=='\\')o<<'\\';o<<c;}o<<'"';}
static void perturb_callback_record(btManifoldPoint& cp,const btCollisionObjectWrapper* a,const btCollisionObjectWrapper* b,int part0,int index0,int part1,int index1){
  if(!g_perturb_callback_context||!g_perturb_callback_context->out||!a||!b)return;
  auto& o=*g_perturb_callback_context->out; auto* wa=dynamic_cast<const CollisionObjectWrapper*>(a->getCollisionObject()); auto* wb=dynamic_cast<const CollisionObjectWrapper*>(b->getCollisionObject());
  o<<"{\"status\":\"callback_observed\",\"backend\":"; perturb_json_string(o,g_perturb_callback_context->backend);
  o<<",\"node_id\":"; perturb_json_string(o,g_perturb_callback_context->node_id); o<<",\"waypoint_id\":"<<g_perturb_callback_context->waypoint_id;
  o<<",\"joint_name\":"; perturb_json_string(o,g_perturb_callback_context->joint_name); o<<",\"delta_rad\":"<<std::setprecision(17)<<g_perturb_callback_context->delta_rad;
  o<<",\"invocation_id\":"; perturb_json_string(o,g_perturb_callback_context->node_id+"_"+g_perturb_callback_context->joint_name+"_"+std::to_string(g_perturb_callback_context->delta_rad)+"_"+std::to_string(g_perturb_callback_context->callback_index++));
  o<<",\"body_name_0\":"; perturb_json_string(o,wa?wa->getName():a->getCollisionShape()->getName()); o<<",\"body_name_1\":"; perturb_json_string(o,wb?wb->getName():b->getCollisionShape()->getName());
  o<<",\"parent_shape_type_0\":"<<a->getCollisionShape()->getShapeType()<<",\"parent_shape_type_1\":"<<b->getCollisionShape()->getShapeType();
  o<<",\"child_index_0\":"<<index0<<",\"child_index_1\":"<<index1<<",\"m_distance1\":"<<std::setprecision(17)<<cp.m_distance1<<",\"m_distance1_hex\":"; perturb_json_string(o,hex_float(cp.m_distance1));
  o<<",\"m_partId0\":"<<part0<<",\"m_partId1\":"<<part1<<",\"m_index0\":"<<index0<<",\"m_index1\":"<<index1<<",\"life_time\":"<<cp.m_lifeTime<<"}\n"; o.flush();
}
using OriginalPerturbAddSingleResult = btScalar (*)(collision_detection_bullet::BroadphaseContactResultCallback*,btManifoldPoint&,const btCollisionObjectWrapper*,int,int,const btCollisionObjectWrapper*,int,int);
static OriginalPerturbAddSingleResult resolve_perturb_original(){
  static OriginalPerturbAddSingleResult fn=nullptr; static bool done=false; if(done)return fn; done=true;
  constexpr const char* symbol="_ZN26collision_detection_bullet31BroadphaseContactResultCallback15addSingleResultER15btManifoldPointPK24btCollisionObjectWrapperiiS5_ii";
  fn=reinterpret_cast<OriginalPerturbAddSingleResult>(dlsym(RTLD_NEXT,symbol));
  if(!fn){void*h=dlopen("/opt/ros/jazzy/lib/libmoveit_collision_detection_bullet.so.2.12.4",RTLD_LAZY|RTLD_LOCAL);if(h)fn=reinterpret_cast<OriginalPerturbAddSingleResult>(dlsym(h,symbol));}
  return fn;
}
namespace collision_detection_bullet {
btScalar BroadphaseContactResultCallback::addSingleResult(btManifoldPoint& cp,const btCollisionObjectWrapper* obj0,int part0,int index0,const btCollisionObjectWrapper* obj1,int part1,int index1){
  perturb_callback_record(cp,obj0,obj1,part0,index0,part1,index1); auto fn=resolve_perturb_original(); return fn?fn(this,cp,obj0,part0,index0,obj1,part1,index1):btScalar(0);
}
}
struct C{int wp;std::string id;std::vector<double>q;};
static std::vector<std::string> split(const std::string&s){std::vector<std::string>o;std::string x;bool q=false;for(char c:s){if(c=='"')q=!q;else if(c==','&&!q){o.push_back(x);x.clear();}else x+=c;}o.push_back(x);return o;}
static double n(const std::string&s){try{return std::stod(s);}catch(...){return 0.;}}
static std::vector<double> vf(std::string s){auto a=s.find('['),b=s.rfind(']');if(a!=std::string::npos)s=s.substr(a+1,b>a?b-a-1:s.size());std::vector<double>o;for(auto&x:split(s))o.push_back(n(x));return o;}
static std::vector<C> candidates(const fs::path&p){std::ifstream f(p);std::string line;std::getline(f,line);auto h=split(line);std::map<std::string,size_t>m;for(size_t i=0;i<h.size();++i)m[h[i]]=i;std::vector<C>o;while(std::getline(f,line)){auto r=split(line);auto q=vf(r[m["joint_values"]]);if(q.size()==6)o.push_back({int(n(r[m["waypoint_id"]])),r[m["candidate_id"]],q});}return o;}
static std::set<std::string> ids(const fs::path&p){std::set<std::string>o;std::ifstream f(p);std::string s;while(std::getline(f,s)){if(!s.empty()&&s.back()=='\r')s.pop_back();if(!s.empty()&&static_cast<unsigned char>(s.front())==0xef)s=s.substr(3);if(!s.empty())o.insert(s);}return o;}
struct MB{shape_msgs::msg::Mesh m;std::map<std::string,uint32_t>i;uint32_t add(double x,double y,double z){std::ostringstream k;k<<std::setprecision(17)<<x<<','<<y<<','<<z;auto a=i.find(k.str());if(a!=i.end())return a->second;geometry_msgs::msg::Point p;p.x=x;p.y=y;p.z=z;auto z0=uint32_t(m.vertices.size());m.vertices.push_back(p);i[k.str()]=z0;return z0;}};
static shape_msgs::msg::Mesh mesh(const fs::path&p){std::ifstream f(p);MB b;std::string l,w;std::vector<std::array<double,3>>t;while(std::getline(f,l)){std::stringstream s(l);s>>w;if(w!="vertex")continue;std::array<double,3>v{};s>>v[0]>>v[1]>>v[2];t.push_back(v);if(t.size()==3){shape_msgs::msg::MeshTriangle q;for(int j=0;j<3;++j)q.vertex_indices[j]=b.add(t[j][0],t[j][1],t[j][2]);b.m.triangles.push_back(q);t.clear();}}return b.m;}
static std::vector<shape_msgs::msg::Mesh> parts(const fs::path&p){std::vector<fs::path>a;for(auto&e:fs::directory_iterator(p))if(e.path().extension()==".stl")a.push_back(e.path());std::sort(a.begin(),a.end());std::vector<shape_msgs::msg::Mesh>o;for(auto&x:a)o.push_back(mesh(x));return o;}
static moveit_msgs::msg::CollisionObject world(const std::vector<shape_msgs::msg::Mesh>&m){moveit_msgs::msg::CollisionObject o;o.header.frame_id="base_link";o.id="horseshoe_collision_compound";o.meshes=m;o.mesh_poses.resize(m.size());for(auto&p:o.mesh_poses){p.orientation.w=1.;p.position.x=.32;p.position.y=-.05;p.position.z=-.38;}o.operation=moveit_msgs::msg::CollisionObject::ADD;return o;}
static void js(std::ostream&o,const std::string&s){o<<'"';for(char c:s){if(c=='"'||c=='\\')o<<'\\';o<<c;}o<<'"';}
int main(int argc,char**argv){rclcpp::init(argc,argv);auto node=std::make_shared<rclcpp::Node>("stage23a7_perturbation_probe",rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));std::string csv,dir,outdir,backend,manifest,group;node->get_parameter_or("candidate_csv",csv,std::string{});node->get_parameter_or("parts_dir",dir,std::string{});node->get_parameter_or("output_dir",outdir,std::string{});node->get_parameter_or("backend",backend,std::string("bullet"));node->get_parameter_or("disputed_manifest",manifest,std::string{});node->get_parameter_or("group_name",group,std::string("fairino5_v6_group"));auto cs=candidates(csv);auto want=ids(manifest);robot_model_loader::RobotModelLoader::Options op("robot_description");op.load_kinematics_solvers=false;robot_model_loader::RobotModelLoader loader(node,op);auto model=loader.getModel();if(!model)return 3;auto scene=std::make_unique<planning_scene::PlanningScene>(model);scene->processCollisionObjectMsg(world(parts(dir)));if(backend=="bullet")scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create());else scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());auto*jmg=model->getJointModelGroup(group);if(!jmg)return 4;fs::create_directories(outdir);std::ofstream out(fs::path(outdir)/(backend+"_joint_perturbation_results.jsonl"));std::ofstream raw(fs::path(outdir)/(backend=="bullet"?"bullet_runtime_manifolds.jsonl":"fcl_runtime_manifolds.jsonl"));collision_detection::CollisionRequest r;r.group_name=group;r.contacts=true;r.max_contacts=524288;r.max_contacts_per_pair=65536;r.pad_environment_collisions=true;r.pad_self_collisions=false;const double d[]={1e-7,1e-6,1e-5,1e-4};for(auto&c:cs)if(want.count(c.id))for(int j=0;j<6;++j)for(double z:d)for(int sg:{-1,1}){auto q=c.q;q[j]+=sg*z;moveit::core::RobotState s(model);s.setVariablePositions(q);bool bounds=s.satisfiesBounds(jmg);s.updateCollisionBodyTransforms();collision_detection::CollisionResult cr;cr.clear();PerturbCallbackContext ctx;ctx.out=&raw;ctx.backend=backend;ctx.node_id=c.id;ctx.waypoint_id=c.wp;ctx.joint_name="j"+std::to_string(j+1);ctx.delta_rad=sg*z;g_perturb_callback_context=(backend=="bullet"?&ctx:nullptr);scene->checkSelfCollision(r,cr,s);g_perturb_callback_context=nullptr;out<<"{\"backend\":";js(out,backend);out<<",\"node_id\":";js(out,c.id);out<<",\"waypoint_id\":"<<c.wp<<",\"joint_name\":\"j"<<(j+1)<<"\",\"delta_rad\":"<<std::setprecision(17)<<(sg*z)<<",\"satisfies_bounds\":"<<(bounds?"true":"false")<<",\"collision\":"<<(cr.collision?"true":"false")<<",\"contact_count\":";size_t count=0;for(auto&kv:cr.contacts)count+=kv.second.size();out<<count<<",\"from_original_frozen_state\":true,\"independent_min_clearance\":null,\"bullet_m_distance1\":\""<<(backend=="bullet"?"observed_callback_records":"not_applicable_fcl")<<"\"}\n";}rclcpp::shutdown();return 0;}
