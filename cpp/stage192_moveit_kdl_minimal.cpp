#include <geometry_msgs/msg/pose.hpp>
#include <moveit/kinematics_base/kinematics_base.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <srdfdom/srdfdom/model.h>
#include <urdf/model.h>

#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
std::vector<double> read_f64(const std::string& path, std::size_t count) {
  std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<double> v(count); in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(double)));
  if (in.gcount() != static_cast<std::streamsize>(count * sizeof(double))) throw std::runtime_error("short binary input"); return v;
}
std::string json_array(const std::vector<double>& v) { std::ostringstream o; o << "[" << std::setprecision(17); for (std::size_t i=0;i<v.size();++i){if(i)o<<",";o<<v[i];} o<<"]"; return o.str(); }
struct Args { std::string urdf, srdf, target, seed, output, mode="same_instance"; int repeat=100; };
Args parse(int argc,char** argv){Args a;for(int i=1;i+1<argc;i+=2){std::string k(argv[i]),v(argv[i+1]);if(k=="--urdf")a.urdf=v;else if(k=="--srdf")a.srdf=v;else if(k=="--target")a.target=v;else if(k=="--seed")a.seed=v;else if(k=="--output")a.output=v;else if(k=="--mode")a.mode=v;else if(k=="--repeat")a.repeat=std::stoi(v);}if(a.urdf.empty()||a.srdf.empty()||a.target.empty()||a.seed.empty()||a.output.empty())throw std::runtime_error("missing C++ MoveIt arguments");return a;}
}

int main(int argc,char** argv) {
  try {
    const Args args=parse(argc,argv); const auto m=read_f64(args.target,16); const auto seed=read_f64(args.seed,6);
    auto urdf=std::make_shared<urdf::Model>(); if(!urdf->initFile(args.urdf))throw std::runtime_error("URDF parse failed");
    auto srdf=std::make_shared<srdf::Model>(); if(!srdf->initFile(*urdf,args.srdf))throw std::runtime_error("SRDF parse failed");
    rclcpp::init(argc,argv); auto node=std::make_shared<rclcpp::Node>("stage192_cpp_moveit_kdl");
    auto model=std::make_shared<moveit::core::RobotModel>(urdf,srdf);
    pluginlib::ClassLoader<kinematics::KinematicsBase> loader("moveit_core","kinematics::KinematicsBase");
    auto create_solver=[&](){auto solver=loader.createSharedInstance("kdl_kinematics_plugin/KDLKinematicsPlugin"); if(!solver->initialize(node,*model,"fairino5_v6_group","base_link",{"spray_tcp_link"},0.005)) throw std::runtime_error("KDL plugin initialize failed"); return solver;};
    auto shared=(args.mode=="same_instance")?create_solver():kinematics::KinematicsBasePtr();
    geometry_msgs::msg::Pose pose; pose.position.x=m[3]; pose.position.y=m[7]; pose.position.z=m[11];
    // The snapshot stores the exact target transform; reconstruct a normalized quaternion for the ROS API.
    const double tr=m[0]+m[5]+m[10]; double qx,qy,qz,qw;
    if (tr > 0.0) { const double s=std::sqrt(tr+1.0)*2.0; qw=0.25*s; qx=(m[9]-m[6])/s; qy=(m[2]-m[8])/s; qz=(m[4]-m[1])/s; }
    else if (m[0] > m[5] && m[0] > m[10]) { const double s=std::sqrt(1.0+m[0]-m[5]-m[10])*2.0; qw=(m[9]-m[6])/s; qx=0.25*s; qy=(m[1]+m[4])/s; qz=(m[2]+m[8])/s; }
    else if (m[5] > m[10]) { const double s=std::sqrt(1.0+m[5]-m[0]-m[10])*2.0; qw=(m[2]-m[8])/s; qx=(m[1]+m[4])/s; qy=0.25*s; qz=(m[6]+m[9])/s; }
    else { const double s=std::sqrt(1.0+m[10]-m[0]-m[5])*2.0; qw=(m[4]-m[1])/s; qx=(m[2]+m[8])/s; qy=(m[6]+m[9])/s; qz=0.25*s; }
    pose.orientation.x=qx;pose.orientation.y=qy;pose.orientation.z=qz;pose.orientation.w=qw;
    std::ofstream out(args.output); out<<"{\"schema_version\":\"1.0\",\"backend\":\"cpp_moveit_kdl_plugin\",\"mode\":\""<<args.mode<<"\",\"repeat\":"<<args.repeat<<",\"records\":["; bool first=true;
    for(int i=0;i<args.repeat;++i){auto solver=shared?shared:create_solver();std::vector<double> solution;moveit_msgs::msg::MoveItErrorCodes error;const bool ok=solver->getPositionIK(pose,seed,solution,error);moveit::core::RobotState state(model);if(ok)state.setJointGroupPositions("fairino5_v6_group",solution);state.update();double fk=-1.0;if(ok){const auto t=state.getGlobalLinkTransform("spray_tcp_link");fk=(t.translation()-Eigen::Vector3d(m[3],m[7],m[11])).norm();}if(!first)out<<",";first=false;out<<"{\"iteration\":"<<i<<",\"process_id\":"<<static_cast<long long>(::getpid())<<",\"solver_instance_id\":\"cpp_moveit_kdl:"<<args.mode<<":"<<(args.mode=="same_instance"?1:i+1)<<"\",\"solver_success\":"<<(ok?"true":"false")<<",\"solver_error_code\":"<<error.val<<",\"solution\":"<<json_array(solution)<<",\"fk_position_error_m\":"<<std::setprecision(17)<<fk<<"}";if(!shared)solver.reset();}
    out<<"]}\n"; shared.reset(); rclcpp::shutdown(); return 0;
  } catch(const std::exception& e){std::cerr<<"stage192_moveit_kdl_minimal: "<<e.what()<<"\n";if(rclcpp::ok())rclcpp::shutdown();return 2;}
}
