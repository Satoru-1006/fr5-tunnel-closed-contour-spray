#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection/collision_env.hpp>
#include <moveit/collision_detection_bullet/collision_detector_allocator_bullet.hpp>
#include <moveit/collision_detection_fcl/collision_detector_allocator_fcl.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <srdfdom/srdfdom/model.h>
#include <urdf/model.h>

namespace fs = std::filesystem;
using moveit::core::RobotState;

struct Args {
  std::string poses, cases, urdf, srdf, output;
  std::string group{"fairino5_v6_group"};
  std::string tip{"spray_tcp_link"};
};

struct PoseRow { Eigen::Vector3d p{Eigen::Vector3d::Zero()}; Eigen::Quaterniond q{Eigen::Quaterniond::Identity()}; Eigen::Vector3d n{Eigen::Vector3d::UnitZ()}; };
struct Case { std::string id, path, family; };

static std::vector<std::string> csv(const std::string& line)
{
  std::vector<std::string> out; std::string item; std::stringstream ss(line);
  while (std::getline(ss, item, ',')) { if (!item.empty() && item.back() == '\r') item.pop_back(); out.push_back(item); }
  if (!line.empty() && line.back() == ',') out.emplace_back();
  return out;
}
static std::string q(const std::string& value)
{
  std::ostringstream out; out << '"';
  for (const unsigned char c : value) { if (c == '"') out << "\\\""; else if (c == '\\') out << "\\\\"; else if (c == '\n') out << "\\n"; else out << c; }
  out << '"'; return out.str();
}
static std::string num(double v)
{ if (!std::isfinite(v)) return "null"; std::ostringstream out; out << std::setprecision(17) << v; return out.str(); }
static std::string arr(const std::vector<double>& values)
{ std::ostringstream out; out << '['; for (size_t i=0; i<values.size(); ++i) { if (i) out << ','; out << num(values[i]); } out << ']'; return out.str(); }
static std::string arr3(const Eigen::Vector3d& v)
{ return "[" + num(v.x()) + "," + num(v.y()) + "," + num(v.z()) + "]"; }
static double d(const std::string& s) { return std::stod(s); }

static Args parse_args(int argc, char** argv)
{
  Args a;
  for (int i=1; i+1<argc; i+=2) {
    const std::string k(argv[i]), v(argv[i+1]);
    if (k=="--poses") a.poses=v; else if (k=="--cases") a.cases=v; else if (k=="--urdf") a.urdf=v;
    else if (k=="--srdf") a.srdf=v; else if (k=="--output") a.output=v; else if (k=="--group") a.group=v; else if (k=="--tip") a.tip=v;
  }
  if (a.poses.empty() || a.cases.empty() || a.urdf.empty() || a.srdf.empty() || a.output.empty()) throw std::runtime_error("missing --poses/--cases/--urdf/--srdf/--output");
  return a;
}

static std::vector<PoseRow> read_poses(const fs::path& path)
{
  std::ifstream in(path); if (!in) throw std::runtime_error("cannot open poses: " + path.string());
  std::string line; std::getline(in,line); std::vector<PoseRow> rows;
  while (std::getline(in,line)) { if (line.empty()) continue; const auto f=csv(line); if (f.size()<10) throw std::runtime_error("invalid pose row"); PoseRow r; r.p=Eigen::Vector3d(d(f[0]),d(f[1]),d(f[2])); r.q=Eigen::Quaterniond(d(f[6]),d(f[3]),d(f[4]),d(f[5])); r.q.normalize(); r.n=Eigen::Vector3d(d(f[7]),d(f[8]),d(f[9])); if (r.n.norm()<1e-9) throw std::runtime_error("zero pose normal"); r.n.normalize(); rows.push_back(r); }
  return rows;
}
static std::vector<Case> read_cases(const fs::path& path)
{
  std::ifstream in(path); if (!in) throw std::runtime_error("cannot open cases: " + path.string());
  std::string line; std::getline(in,line); std::vector<Case> out;
  while (std::getline(in,line)) { if (line.empty()) continue; const auto f=csv(line); if (f.size()<3) throw std::runtime_error("invalid case row"); out.push_back({f[0],f[1],f[2]}); }
  if (out.empty()) throw std::runtime_error("case manifest is empty"); return out;
}
static std::vector<std::vector<double>> read_joint_csv(const fs::path& path)
{
  std::ifstream in(path); if (!in) throw std::runtime_error("cannot open trajectory: " + path.string());
  std::string line; if (!std::getline(in,line)) throw std::runtime_error("empty trajectory"); std::vector<std::vector<double>> rows;
  while (std::getline(in,line)) { if (line.empty()) continue; const auto f=csv(line); if (f.size()<7) throw std::runtime_error("trajectory row has fewer than 6 joints"); std::vector<double> r; for (size_t i=f.size()-6;i<f.size();++i) r.push_back(d(f[i])); rows.push_back(std::move(r)); }
  if (rows.empty()) throw std::runtime_error("trajectory has no states"); return rows;
}

static moveit_msgs::msg::CollisionObject box(const std::string& id, const Eigen::Vector3d& center, const Eigen::Vector3d& dims, const Eigen::Quaterniond& orientation)
{
  moveit_msgs::msg::CollisionObject obj; obj.header.frame_id="base_link"; obj.id=id;
  shape_msgs::msg::SolidPrimitive primitive; primitive.type=shape_msgs::msg::SolidPrimitive::BOX; primitive.dimensions={dims.x(),dims.y(),dims.z()};
  geometry_msgs::msg::Pose pose; pose.position.x=center.x(); pose.position.y=center.y(); pose.position.z=center.z(); pose.orientation.x=orientation.x(); pose.orientation.y=orientation.y(); pose.orientation.z=orientation.z(); pose.orientation.w=orientation.w();
  obj.primitives.push_back(primitive); obj.primitive_poses.push_back(pose); obj.operation=moveit_msgs::msg::CollisionObject::ADD; return obj;
}
static std::unique_ptr<planning_scene::PlanningScene> make_world(const moveit::core::RobotModelConstPtr& model, const std::vector<PoseRow>& poses, bool floor, bool bullet)
{
  auto scene=std::make_unique<planning_scene::PlanningScene>(model);
  for (size_t i=0;i+1<poses.size();++i) {
    const Eigen::Vector3d p0=poses[i].p+0.260*poses[i].n, p1=poses[i+1].p+0.260*poses[i+1].n; Eigen::Vector3d x=p1-p0; const double length=x.norm(); if (length<1e-10) continue; x/=length;
    Eigen::Vector3d y=Eigen::Vector3d::UnitY(); Eigen::Vector3d z=x.cross(y); if (z.norm()<1e-9) z=poses[i].n; else z.normalize(); y=z.cross(x).normalized(); Eigen::Matrix3d R; R.col(0)=x; R.col(1)=y; R.col(2)=z;
    scene->processCollisionObjectMsg(box("horseshoe_wall_"+std::to_string(i), (p0+p1)/2.0+0.020*poses[i].n, Eigen::Vector3d(length+0.040,1.10,0.040), Eigen::Quaterniond(R)));
  }
  if (floor) { double xmin=std::numeric_limits<double>::infinity(), xmax=-xmin, ymin=0.0; for (const auto& p:poses) { xmin=std::min(xmin,p.p.x()); xmax=std::max(xmax,p.p.x()); ymin+=p.p.y(); } ymin/=static_cast<double>(poses.size()); scene->processCollisionObjectMsg(box("tunnel_floor", Eigen::Vector3d((xmin+xmax)/2.0,ymin,-0.220), Eigen::Vector3d(xmax-xmin+0.040,1.10,0.040), Eigen::Quaterniond::Identity())); }
  if (bullet) scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create());
  else scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());
  return scene;
}
static collision_detection::CollisionRequest request(const std::string& group, bool padded)
{ collision_detection::CollisionRequest r; r.group_name=group; r.contacts=true; r.max_contacts=256; r.max_contacts_per_pair=8; r.pad_environment_collisions=padded; r.pad_self_collisions=false; r.distance=false; return r; }
static collision_detection::CollisionResult world(const collision_detection::CollisionEnv* env, const collision_detection::CollisionRequest& r, const RobotState& s, const collision_detection::AllowedCollisionMatrix& acm)
{ collision_detection::CollisionResult x; x.clear(); env->checkRobotCollision(r,x,s,acm); return x; }
static collision_detection::CollisionResult ccd(const collision_detection::CollisionEnv* env, const collision_detection::CollisionRequest& r, const RobotState& a, const RobotState& b, const collision_detection::AllowedCollisionMatrix& acm)
{ collision_detection::CollisionResult x; x.clear(); env->checkRobotCollision(r,x,a,b,acm); return x; }
static collision_detection::CollisionResult self(const planning_scene::PlanningScene& scene, const collision_detection::CollisionRequest& r, const RobotState& s, const collision_detection::AllowedCollisionMatrix& acm)
{ collision_detection::CollisionResult x; x.clear(); scene.checkSelfCollision(r,x,s,acm); return x; }
static std::vector<std::string> pairs(const collision_detection::CollisionResult& result)
{ std::vector<std::string> out; for (const auto& c:result.contacts) out.push_back(c.first.first+"|"+c.first.second); std::sort(out.begin(),out.end()); out.erase(std::unique(out.begin(),out.end()),out.end()); return out; }
static std::string pairs_json(const collision_detection::CollisionResult& result)
{ const auto p=pairs(result); std::ostringstream out; out<<'['; for(size_t i=0;i<p.size();++i){if(i)out<<',';out<<q(p[i]);}out<<']';return out.str(); }

struct Distance { bool available{false}; double value{std::numeric_limits<double>::quiet_NaN()}; std::string pair; };
static Distance distance(const collision_detection::CollisionEnv* env, const std::string& group, const collision_detection::AllowedCollisionMatrix& acm, const RobotState& s, bool self_query)
{
  collision_detection::DistanceRequest r; r.type=collision_detection::DistanceRequestTypes::GLOBAL; r.group_name=group; r.enable_nearest_points=true; r.enable_signed_distance=true; r.acm=&acm; collision_detection::DistanceResult x; if(self_query)env->distanceSelf(r,x,s);else env->distanceRobot(r,x,s);
  const double sentinel=std::numeric_limits<double>::max()*0.5; Distance out; out.available=std::isfinite(x.minimum_distance.distance)&&std::abs(x.minimum_distance.distance)<sentinel; if(out.available){out.value=x.minimum_distance.distance; out.pair=x.minimum_distance.link_names[0]+"|"+x.minimum_distance.link_names[1];} return out;
}

static void write_jacobian(std::ofstream& out, std::ofstream& full_out, std::ofstream& vectors_out, std::ofstream& split_out, const std::string& id, int waypoint, const RobotState& state, const moveit::core::JointModelGroup* group, const moveit::core::LinkModel* tip, double& min_sigma, double& max_cond, int& min_wp)
{
  Eigen::MatrixXd J, U, V; const bool ok=state.getJacobian(group,tip,Eigen::Vector3d::Zero(),J,false); Eigen::VectorXd sv, translational_sv, rotational_sv; double sigma_min=std::numeric_limits<double>::quiet_NaN(), sigma_max=std::numeric_limits<double>::quiet_NaN(), cond=std::numeric_limits<double>::quiet_NaN(), manip=std::numeric_limits<double>::quiet_NaN(); int rank=0;
  if(ok&&J.rows()>0&&J.cols()>0){Eigen::JacobiSVD<Eigen::MatrixXd> s(J,Eigen::ComputeFullU|Eigen::ComputeFullV);sv=s.singularValues();U=s.matrixU();V=s.matrixV();if(sv.size()){sigma_max=sv(0);sigma_min=sv(sv.size()-1);const double tol=sigma_max*std::max(J.rows(),J.cols())*std::numeric_limits<double>::epsilon()*100.0;for(int i=0;i<sv.size();++i)if(sv(i)>tol)++rank;cond=sigma_min>0?sigma_max/sigma_min:std::numeric_limits<double>::infinity();manip=sv.prod();if(std::isfinite(sigma_min)&&sigma_min<min_sigma){min_sigma=sigma_min;min_wp=waypoint;}if(std::isfinite(cond)&&cond>max_cond)max_cond=cond;}if(J.rows()>=3){translational_sv=J.topRows(3).jacobiSvd().singularValues();rotational_sv=J.bottomRows(3).jacobiSvd().singularValues();}}
  out<<q(id)<<','<<waypoint<<','<<(ok?J.rows():0); for(int i=0;i<6;++i)out<<','<<(i<sv.size()?num(sv(i)):"null"); out<<','<<num(sigma_min)<<','<<num(sigma_max)<<','<<num(cond)<<','<<rank<<','<<num(manip)<<"\n";
  full_out<<q(id)<<','<<waypoint; for(int r=0;r<6;++r)for(int c=0;c<6;++c)full_out<<','<<((ok&&r<J.rows()&&c<J.cols())?num(J(r,c)):"null"); full_out<<"\n";
  vectors_out<<q(id)<<','<<waypoint; for(int i=0;i<6;++i)vectors_out<<','<<((i<U.rows()&&U.cols()>0)?num(U(i,U.cols()-1)):"null"); for(int i=0;i<6;++i)vectors_out<<','<<((i<V.rows()&&V.cols()>0)?num(V(i,V.cols()-1)):"null"); vectors_out<<"\n";
  split_out<<q(id)<<','<<waypoint; for(int i=0;i<3;++i)split_out<<','<<(i<translational_sv.size()?num(translational_sv(i)):"null"); for(int i=0;i<3;++i)split_out<<','<<(i<rotational_sv.size()?num(rotational_sv(i)):"null"); split_out<<"\n";
}

static void run_case(const Case& c, const fs::path& outdir, const std::vector<PoseRow>& nominal_poses, const moveit::core::RobotModelConstPtr& model, const std::string& group_name, const std::string& tip_name, std::ofstream& seg, std::ofstream& clr, std::ofstream& jac, std::ofstream& full_jac, std::ofstream& sv_vectors, std::ofstream& split_jac, std::ofstream& summary)
{
  const auto qs=read_joint_csv(c.path); const auto scene=make_world(model,nominal_poses,true,true); const auto distance_scene=make_world(model,nominal_poses,true,false); const auto& acm=scene->getAllowedCollisionMatrix(); const auto env=scene->getCollisionEnv(); const auto distance_env=distance_scene->getCollisionEnv(); const auto req=request(group_name,true); const auto self_req=request(group_name,false); const auto* jmg=model->getJointModelGroup(group_name); const auto* tip=model->getLinkModel(tip_name); if(!jmg||!tip)throw std::runtime_error("group or tip missing");
  double min_world=std::numeric_limits<double>::infinity(), min_self=std::numeric_limits<double>::infinity(), min_sigma=std::numeric_limits<double>::infinity(), max_cond=0.0; int min_world_wp=-1,min_self_wp=-1,min_sigma_wp=-1,world_collisions=0,self_collisions=0,ccd_collisions=0,self_samples=0; std::string min_world_pair,min_self_pair;
  std::vector<RobotState> states; states.reserve(qs.size()); for(const auto& values:qs){RobotState s(model);s.setJointGroupPositions(jmg,values);s.update();states.push_back(s);}
  for(size_t i=0;i<states.size();++i){const auto& s=states[i]; const auto w=world(env.get(),req,s,acm); const auto se=self(*scene,self_req,s,acm); if(w.collision)++world_collisions;if(se.collision)++self_collisions;const auto wd=distance(distance_env.get(),group_name,acm,s,false);const auto sd=distance(distance_env.get(),group_name,acm,s,true);if(wd.available&&wd.value<min_world){min_world=wd.value;min_world_wp=static_cast<int>(i);min_world_pair=wd.pair;}if(sd.available&&sd.value<min_self){min_self=sd.value;min_self_wp=static_cast<int>(i);min_self_pair=sd.pair;}clr<<q(c.id)<<','<<i<<','<<num(wd.value)<<','<<num(sd.value)<<','<<q(wd.pair)<<','<<q(sd.pair)<<"\n";write_jacobian(jac,full_jac,sv_vectors,split_jac,c.id,static_cast<int>(i),s,jmg,tip,min_sigma,max_cond,min_sigma_wp);}
  for(size_t i=0;i+1<states.size();++i){const auto cc=ccd(env.get(),req,states[i],states[i+1],acm);if(cc.collision)++ccd_collisions;double maxdq=0.0;for(int j=0;j<6;++j)maxdq=std::max(maxdq,std::abs(qs[i+1][j]-qs[i][j]));const int n=std::max(1,static_cast<int>(std::ceil(maxdq/0.008726646259971648)));for(int k=0;k<=n;++k){const double alpha=static_cast<double>(k)/n;std::vector<double> v(6);for(int j=0;j<6;++j)v[j]=qs[i][j]+alpha*(qs[i+1][j]-qs[i][j]);RobotState sample(model);sample.setJointGroupPositions(jmg,v);sample.update();const auto sd=distance(distance_env.get(),group_name,acm,sample,true);const auto ss=self(*scene,self_req,sample,acm);++self_samples;if(sd.available&&sd.value<min_self){min_self=sd.value;min_self_wp=static_cast<int>(i);min_self_pair=sd.pair;}seg<<q(c.id)<<','<<i<<','<<num(alpha)<<','<<(world(env.get(),req,sample,acm).collision?"true":"false")<<','<<(ss.collision?"true":"false")<<','<<(cc.collision?"true":"false")<<','<<pairs_json(cc)<<','<<n<<"\n";}}
  const bool pass=world_collisions==0&&self_collisions==0&&ccd_collisions==0; summary<<"{\"case_id\":"<<q(c.id)<<",\"family\":"<<q(c.family)<<",\"state_count\":"<<qs.size()<<",\"nominal_waypoint_world_collision_count\":"<<world_collisions<<",\"nominal_waypoint_self_collision_count\":"<<self_collisions<<",\"native_continuous_segment_collision_count\":"<<ccd_collisions<<",\"self_adaptive_sample_count\":"<<self_samples<<",\"self_adaptive_step_rad\":0.008726646259971648,\"minimum_robot_world_distance_m\":"<<num(min_world)<<",\"minimum_robot_world_waypoint\":"<<min_world_wp<<",\"minimum_robot_world_pair\":"<<q(min_world_pair)<<",\"minimum_self_distance_m\":"<<num(min_self)<<",\"minimum_self_waypoint\":"<<min_self_wp<<",\"minimum_self_pair\":"<<q(min_self_pair)<<",\"minimum_jacobian_sigma\":"<<num(min_sigma)<<",\"maximum_jacobian_condition_number\":"<<num(max_cond)<<",\"native_continuous_robot_world\":true,\"status\":"<<q(pass?"PASS":"FAIL")<<"}\n";
}

static void control_record(std::ofstream& out,const std::string& name,bool pass,const std::string& detail)
{ out<<"{\"control\":"<<q(name)<<",\"status\":"<<q(pass?"PASS":"FAIL")<<",\"detail\":"<<q(detail)<<"}\n"; }
static std::unique_ptr<planning_scene::PlanningScene> one_box(const moveit::core::RobotModelConstPtr& model,const Eigen::Vector3d& center,double side,bool bullet=true)
{auto s=std::make_unique<planning_scene::PlanningScene>(model);s->processCollisionObjectMsg(box("d41_control_obstacle",center,Eigen::Vector3d::Constant(side),Eigen::Quaterniond::Identity()));if(bullet)s->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create());else s->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorFCL::create());return s;}
static bool find_swept_control(const moveit::core::RobotModelConstPtr& model,const std::string& group,std::ofstream& out,std::vector<double>& fq0,std::vector<double>& fq1,Eigen::Vector3d& fc,double& fs)
{
  const std::vector<std::string> links={"wrist3_link","wrist2_link","forearm_link","upperarm_link","shoulder_link","base_link"}; const std::vector<double> side={0.015,0.025,0.040,0.060,0.100}; std::vector<std::pair<std::vector<double>,std::vector<double>>> pairs;
  for(int j=0;j<6;++j){std::vector<double>a(6),b(6);a[j]=-1.25;b[j]=1.25;pairs.push_back({a,b});a[j]=1.25;b[j]=-1.25;pairs.push_back({a,b});} pairs.push_back({{-1,-.7,.8,0,0,0},{1,.7,-.8,0,0,0}});
  int trials=0; const auto req=request(group,true); const std::vector<double> known_q0={-1.0,-0.7,0.8,0.0,0.0,0.0}, known_q1={1.0,0.7,-0.8,0.0,0.0,0.0};
  for (const auto& known : {std::pair<std::vector<double>,std::vector<double>>(known_q0, known_q1), std::pair<std::vector<double>,std::vector<double>>(known_q1, known_q0)}) { std::vector<double> mid(6); for(int j=0;j<6;++j) mid[j]=.5*(known.first[j]+known.second[j]); RobotState m(model); m.setJointGroupPositions(model->getJointModelGroup(group),mid); m.updateCollisionBodyTransforms(); const std::string link="forearm_link"; const auto* lm=model->getLinkModel(link); if(lm && !lm->getShapes().empty()) { const Eigen::Vector3d center=m.getCollisionBodyTransform(link,0).translation(); for(double s:{0.040,0.025,0.060}) { ++trials; auto scene=one_box(model,center,s); const auto& acm=scene->getAllowedCollisionMatrix(); const auto env=scene->getCollisionEnv(); RobotState a(model),b(model),mm(model); a.setJointGroupPositions(model->getJointModelGroup(group),known.first); b.setJointGroupPositions(model->getJointModelGroup(group),known.second); mm.setJointGroupPositions(model->getJointModelGroup(group),mid); a.updateCollisionBodyTransforms(); b.updateCollisionBodyTransforms(); mm.updateCollisionBodyTransforms(); const auto aa=world(env.get(),req,a,acm),bb=world(env.get(),req,b,acm),md=world(env.get(),req,mm,acm),cc=ccd(env.get(),req,a,b,acm); if(!aa.collision&&!bb.collision&&md.collision&&cc.collision) { out<<"{\"control\":\"endpoint_free_mid_segment_collision\",\"status\":\"PASS\",\"trials\":"<<trials<<",\"link\":\"forearm_link\",\"obstacle_center\":"<<arr3(center)<<",\"obstacle_side_m\":"<<num(s)<<",\"q0\":"<<arr(known.first)<<",\"q1\":"<<arr(known.second)<<",\"discrete_mid_collision\":true,\"native_continuous_collision\":true}\n"; fq0=known.first; fq1=known.second; fc=center; fs=s; return true; } } } }
  for(const auto& ab:pairs){std::vector<double> mid(6);for(int j=0;j<6;++j)mid[j]=.5*(ab.first[j]+ab.second[j]);RobotState m(model);m.setJointGroupPositions(model->getJointModelGroup(group),mid);m.updateCollisionBodyTransforms();for(const auto& link:links){const auto* lm=model->getLinkModel(link);if(!lm||lm->getShapes().empty())continue;const Eigen::Vector3d center=m.getCollisionBodyTransform(link,0).translation();for(double s:side){++trials;auto scene=one_box(model,center,s);const auto& acm=scene->getAllowedCollisionMatrix();const auto env=scene->getCollisionEnv();RobotState a(model),b(model),mm(model);a.setJointGroupPositions(model->getJointModelGroup(group),ab.first);b.setJointGroupPositions(model->getJointModelGroup(group),ab.second);mm.setJointGroupPositions(model->getJointModelGroup(group),mid);a.updateCollisionBodyTransforms();b.updateCollisionBodyTransforms();mm.updateCollisionBodyTransforms();const auto aa=world(env.get(),req,a,acm),bb=world(env.get(),req,b,acm),md=world(env.get(),req,mm,acm),cc=ccd(env.get(),req,a,b,acm);if(!aa.collision&&!bb.collision&&md.collision&&cc.collision){out<<"{\"control\":\"endpoint_free_mid_segment_collision\",\"status\":\"PASS\",\"trials\":"<<trials<<",\"link\":"<<q(link)<<",\"obstacle_center\":"<<arr3(center)<<",\"obstacle_side_m\":"<<num(s)<<",\"q0\":"<<arr(ab.first)<<",\"q1\":"<<arr(ab.second)<<",\"discrete_mid_collision\":true,\"native_continuous_collision\":true}\n";fq0=ab.first;fq1=ab.second;fc=center;fs=s;return true;}}}}
  out<<"{\"control\":\"endpoint_free_mid_segment_collision\",\"status\":\"FAIL\",\"trials\":"<<trials<<",\"reason\":\"no_native_bullet_ccd_positive_control_found\"}\n";return false;
}
static void run_controls(const moveit::core::RobotModelConstPtr& model,const std::string& group,const fs::path& outdir)
{
  std::ofstream out(outdir/"D41_native_adversarial_controls.jsonl");std::vector<double> q0,q1;Eigen::Vector3d center;double side=0;const bool a=find_swept_control(model,group,out,q0,q1,center,side);if(!a){control_record(out,"free_space_negative_control",false,"dependent swept-positive control unavailable");control_record(out,"self_collision_negative_control",false,"dependent control execution continued but positive world control absent");return;}
  const auto req=request(group,true);auto free_scene=one_box(model,Eigen::Vector3d(100,100,100),side);const auto& acm=free_scene->getAllowedCollisionMatrix();const auto env=free_scene->getCollisionEnv();RobotState s(model),e(model);s.setJointGroupPositions(model->getJointModelGroup(group),q0);e.setJointGroupPositions(model->getJointModelGroup(group),q1);s.updateCollisionBodyTransforms();e.updateCollisionBodyTransforms();const auto fs0=world(env.get(),req,s,acm),fs1=world(env.get(),req,e,acm),fsc=ccd(env.get(),req,s,e,acm);control_record(out,"free_space_negative_control",!fs0.collision&&!fs1.collision&&!fsc.collision,"far obstacle remains free for endpoints and native sweep");
  auto self_scene=std::make_unique<planning_scene::PlanningScene>(model);self_scene->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorBullet::create());bool found=false;std::vector<double> fq;std::vector<std::string> fp;std::vector<double> values={-3.0,-2.5,-2.0,-1.5,-1.0,0.0,1.0,1.5,2.0,2.5,3.0};for(double a0:values)for(double a1:values)if(!found){fq={a0,a1,0,0,0,0};RobotState x(model);x.setJointGroupPositions(model->getJointModelGroup(group),fq);x.updateCollisionBodyTransforms();const auto r=self(*self_scene,request(group,false),x,self_scene->getAllowedCollisionMatrix());if(r.collision){found=true;fp=pairs(r);}}
  if(!found){std::mt19937 rng(41041);std::uniform_real_distribution<double> unit(0.01,0.99);const auto* jmg=model->getJointModelGroup(group);const auto& names=jmg->getVariableNames();for(int trial=0;trial<20000&&!found;++trial){fq.resize(6);for(int j=0;j<6;++j){const auto& b=model->getVariableBounds(names[j]);fq[j]=b.position_bounded_?b.min_position_+unit(rng)*(b.max_position_-b.min_position_):0.0;}RobotState x(model);x.setJointGroupPositions(jmg,fq);x.updateCollisionBodyTransforms();const auto r=self(*self_scene,request(group,false),x,self_scene->getAllowedCollisionMatrix());if(r.collision){found=true;fp=pairs(r);}}}
  control_record(out,"self_collision_negative_control",found,found?"deliberate feasible joint state produces native self collision":"deterministic feasible candidate search found no self collision");if(found)out<<"{\"control\":\"self_collision_negative_control_detail\",\"q\":"<<arr(fq)<<",\"pairs\":";if(found){out<<'[';for(size_t i=0;i<fp.size();++i){if(i)out<<',';out<<q(fp[i]);}out<<"]}\n";}
  bool near=false,far=false;Eigen::Vector3d nearc,far_c;double near_d=0,far_d=0;const std::vector<Eigen::Vector3d> axes={Eigen::Vector3d::UnitX(),-Eigen::Vector3d::UnitX(),Eigen::Vector3d::UnitY(),-Eigen::Vector3d::UnitY(),Eigen::Vector3d::UnitZ(),-Eigen::Vector3d::UnitZ()};RobotState mid(model);std::vector<double> qm(6);for(int j=0;j<6;++j)qm[j]=.5*(q0[j]+q1[j]);mid.setJointGroupPositions(model->getJointModelGroup(group),qm);mid.updateCollisionBodyTransforms();const Eigen::Vector3d c=center;for(const auto& axis:axes)for(double off:{.005,.01,.02,.03,.05,.08,.12,.20})for(double sz:{.005,.01,.02}){auto sc=one_box(model,c+axis*off,sz,true);auto distance_sc=one_box(model,c+axis*off,sz,false);const auto& a2=sc->getAllowedCollisionMatrix();const auto en=sc->getCollisionEnv();const auto den=distance_sc->getCollisionEnv();const auto cc=world(en.get(),req,mid,a2);const auto di=distance(den.get(),group,a2,mid,false);if(!cc.collision&&di.available){if(di.value>0&&di.value<.010&&!near){near=true;nearc=c+axis*off;near_d=di.value;}if(di.value>.050&&!far){far=true;far_c=c+axis*off;far_d=di.value;}}}
  control_record(out,"near_miss_positive_clearance",near,near?"native distance is positive and below 10 mm":"search did not find a positive sub-10-mm native distance");if(near)out<<"{\"control\":\"near_miss_detail\",\"distance_m\":"<<num(near_d)<<",\"obstacle_center\":"<<arr3(nearc)<<"}\n";control_record(out,"comfortable_clearance",far,far?"native distance exceeds 50 mm":"search did not find a positive distance above 50 mm");if(far)out<<"{\"control\":\"comfortable_clearance_detail\",\"distance_m\":"<<num(far_d)<<",\"obstacle_center\":"<<arr3(far_c)<<"}\n";
}

int main(int argc,char** argv)
{
  try {
    const Args a=parse_args(argc,argv); fs::create_directories(a.output);
    const auto poses=read_poses(a.poses); const auto cases=read_cases(a.cases); rclcpp::init(argc,argv);
    auto urdf=std::make_shared<urdf::Model>(); if(!urdf->initFile(a.urdf)) throw std::runtime_error("URDF parse failed");
    auto srdf=std::make_shared<srdf::Model>(); if(!srdf->initFile(*urdf,a.srdf)) throw std::runtime_error("SRDF parse failed");
    auto model=std::make_shared<moveit::core::RobotModel>(urdf,srdf); if(!model->hasJointModelGroup(a.group)) throw std::runtime_error("group missing");
    std::ofstream seg(fs::path(a.output)/"D41_native_segment_collision.csv"),clr(fs::path(a.output)/"D41_native_clearance.csv"),jac(fs::path(a.output)/"D41_native_jacobian.csv"),full_jac(fs::path(a.output)/"D41_native_full_jacobian.csv"),sv_vectors(fs::path(a.output)/"D41_native_svd_vectors.csv"),split_jac(fs::path(a.output)/"D41_native_jacobian_split.csv"),summary(fs::path(a.output)/"D41_native_case_summary.jsonl");
    seg<<"case_id,segment_index,alpha,discrete_sample_world_collision,discrete_sample_self_collision,native_continuous_segment_collision,continuous_contact_pairs,adaptive_subdivision_count\n";
    clr<<"case_id,waypoint,robot_world_distance_m,self_distance_m,robot_world_pair,self_pair\n";
    jac<<"case_id,waypoint,task_dimension,sigma_1,sigma_2,sigma_3,sigma_4,sigma_5,sigma_6,sigma_min,sigma_max,condition_number,effective_rank,manipulability\n";
    full_jac<<"case_id,waypoint"; for(int r=0;r<6;++r) for(int c=0;c<6;++c) full_jac<<",J"<<r<<c; full_jac<<"\n";
    sv_vectors<<"case_id,waypoint,U_sigma_min_0,U_sigma_min_1,U_sigma_min_2,U_sigma_min_3,U_sigma_min_4,U_sigma_min_5,V_sigma_min_0,V_sigma_min_1,V_sigma_min_2,V_sigma_min_3,V_sigma_min_4,V_sigma_min_5\n";
    split_jac<<"case_id,waypoint,trans_sigma_1,trans_sigma_2,trans_sigma_3,rot_sigma_1,rot_sigma_2,rot_sigma_3\n";
    for(const auto& c:cases) run_case(c,a.output,poses,model,a.group,a.tip,seg,clr,jac,full_jac,sv_vectors,split_jac,summary);
    if(!std::getenv("D41_SKIP_CONTROLS")) run_controls(model,a.group,a.output);
    auto final_scene=make_world(model,poses,true,true); std::ofstream prov(fs::path(a.output)/"D41_native_provenance.json");
    prov<<"{\n  \"schema_version\":\"d41-native-provenance-v1\",\n  \"backend\":\"MoveIt2 native CollisionEnvBullet plus CollisionEnvFCL distance\",\n  \"collision_detector\":\"Bullet\",\n  \"continuous_robot_world_api\":\"CollisionEnv::checkRobotCollision(req,result,state1,state2,acm)\",\n  \"distance_robot_api\":\"CollisionEnvFCL::distanceRobot\",\n  \"distance_self_api\":\"CollisionEnvFCL::distanceSelf\",\n  \"self_continuous_api\":\"not_available_in_MoveIt_Bullet_wrapper; adaptive_discrete_refinement_used\",\n  \"full_jacobian_output\":\"D41_native_full_jacobian.csv\",\n  \"svd_direction_output\":\"D41_native_svd_vectors.csv\",\n  \"jacobian_split_output\":\"D41_native_jacobian_split.csv\",\n  \"task_weighted_jacobian\":\"not_applied_without_physical_translation_rotation_weight\",\n  \"moveit_collision_detector_name\":"<<q(final_scene->getCollisionDetectorName())<<",\n  \"world_object_count\":"<<((poses.size()>0?poses.size()-1:0)+1)<<",\n  \"pose_count\":"<<poses.size()<<",\n  \"source_scope\":\"181-point ON-state open-arch only\"\n}\n";
    rclcpp::shutdown(); return 0;
  } catch(const std::exception& e) { std::cerr<<"stage3_h13_d41_native: "<<e.what()<<"\n"; if(rclcpp::ok()) rclcpp::shutdown(); return 2; }
}
