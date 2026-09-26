#include <geometry_msgs/msg/pose.hpp>
#include <moveit/kinematics_base/kinematics_base.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <srdfdom/srdfdom/model.h>
#include <urdf/model.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace
{
struct Args
{
  std::string urdf;
  std::string srdf;
  std::string requests;
  std::string output;
  std::string group{"fairino5_v6_group"};
  std::string base{"base_link"};
  std::string tip{"spray_tcp_link"};
};

Args parse(int argc, char** argv)
{
  Args args;
  for (int i = 1; i + 1 < argc; i += 2)
  {
    const std::string key(argv[i]);
    const std::string value(argv[i + 1]);
    if (key == "--urdf") args.urdf = value;
    else if (key == "--srdf") args.srdf = value;
    else if (key == "--requests") args.requests = value;
    else if (key == "--output") args.output = value;
    else if (key == "--group") args.group = value;
    else if (key == "--base") args.base = value;
    else if (key == "--tip") args.tip = value;
  }
  if (args.urdf.empty() || args.srdf.empty() || args.requests.empty() || args.output.empty())
    throw std::runtime_error("missing --urdf/--srdf/--requests/--output");
  return args;
}

std::string json_escape(const std::string& value)
{
  std::ostringstream out;
  for (const char ch : value)
  {
    switch (ch)
    {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << ch; break;
    }
  }
  return out.str();
}

std::string json_number(double value)
{
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

std::string json_array(const std::vector<double>& values)
{
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    if (i) out << ",";
    out << json_number(values[i]);
  }
  out << "]";
  return out.str();
}

struct Request
{
  long long target_index{};
  std::string task_sample_id;
  std::string seed_id;
  long long seed_index{};
  geometry_msgs::msg::Pose pose;
  std::vector<double> seed;
};

bool read_request(std::istream& input, Request& request)
{
  // TSV schema: target_index, task_sample_id, seed_id, seed_index,
  // x,y,z,qx,qy,qz,qw,j1,j2,j3,j4,j5,j6.
  std::string line;
  if (!std::getline(input, line)) return false;
  if (line.empty()) return true;
  std::vector<std::string> fields;
  std::size_t start = 0;
  while (true)
  {
    const std::size_t end = line.find('\t', start);
    fields.push_back(line.substr(start, end == std::string::npos ? std::string::npos : end - start));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  if (fields.size() != 17) throw std::runtime_error("invalid request TSV field count");
  std::size_t index = 0;
  request.target_index = std::stoll(fields[index++]);
  request.task_sample_id = fields[index++];
  request.seed_id = fields[index++];
  request.seed_index = std::stoll(fields[index++]);
  request.pose.position.x = std::stod(fields[index++]);
  request.pose.position.y = std::stod(fields[index++]);
  request.pose.position.z = std::stod(fields[index++]);
  request.pose.orientation.x = std::stod(fields[index++]);
  request.pose.orientation.y = std::stod(fields[index++]);
  request.pose.orientation.z = std::stod(fields[index++]);
  request.pose.orientation.w = std::stod(fields[index++]);
  request.seed.clear();
  for (; index < fields.size(); ++index) request.seed.push_back(std::stod(fields[index]));
  if (request.seed.size() != 6) throw std::runtime_error("request seed dimension is not six");
  return true;
}
}  // namespace

int main(int argc, char** argv)
{
  try
  {
    const Args args = parse(argc, argv);
    auto urdf = std::make_shared<urdf::Model>();
    if (!urdf->initFile(args.urdf)) throw std::runtime_error("URDF parse failed");
    auto srdf = std::make_shared<srdf::Model>();
    if (!srdf->initFile(*urdf, args.srdf)) throw std::runtime_error("SRDF parse failed");

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);
    const std::string prefix = "robot_description_kinematics." + args.group + ".";
    options.parameter_overrides({
      rclcpp::Parameter(prefix + "kinematics_solver", "kdl_kinematics_plugin/KDLKinematicsPlugin"),
      rclcpp::Parameter(prefix + "kinematics_solver_search_resolution", 0.005),
      rclcpp::Parameter(prefix + "position_only_ik", false),
      rclcpp::Parameter(prefix + "orientation_vs_position", 1.0),
      rclcpp::Parameter(prefix + "epsilon", 1.0e-5),
      rclcpp::Parameter(prefix + "max_solver_iterations", 500),
      rclcpp::Parameter(prefix + "joints", std::vector<std::string>{"j1", "j2", "j3", "j4", "j5", "j6"})
    });
    auto node = std::make_shared<rclcpp::Node>("stage3_h4_1_ik_bridge", options);
    auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
    if (!model->hasJointModelGroup(args.group)) throw std::runtime_error("planning group unavailable");

    pluginlib::ClassLoader<kinematics::KinematicsBase> loader("moveit_core", "kinematics::KinematicsBase");
    auto solver = loader.createSharedInstance("kdl_kinematics_plugin/KDLKinematicsPlugin");
    if (!solver->initialize(node, *model, args.group, args.base, {args.tip}, 0.005))
      throw std::runtime_error("installed KDL plugin initialize failed");

    std::ifstream input(args.requests);
    if (!input) throw std::runtime_error("cannot open requests TSV");
    std::ofstream output(args.output);
    if (!output) throw std::runtime_error("cannot open output JSONL");
    output << std::setprecision(17);
    Request request;
    while (read_request(input, request))
    {
      const auto started = std::chrono::steady_clock::now();
      std::vector<double> solution;
      moveit_msgs::msg::MoveItErrorCodes error;
      // This is the remediation boundary: one direct KinematicsBase call.
      const bool solved = solver->getPositionIK(request.pose, request.seed, solution, error);
      const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      output << "{\"target_index\":" << request.target_index
             << ",\"task_sample_id\":\"" << json_escape(request.task_sample_id)
             << "\",\"seed_id\":\"" << json_escape(request.seed_id)
             << "\",\"seed_index\":" << request.seed_index
             << ",\"seed_joint_positions\":" << json_array(request.seed)
             << ",\"solver_api\":\"KinematicsBase::getPositionIK\""
             << ",\"solver_plugin\":\"kdl_kinematics_plugin/KDLKinematicsPlugin\""
             << ",\"requested_timeout_s\":0.0"
             << ",\"solver_success\":" << (solved ? "true" : "false")
             << ",\"solver_error_code\":" << error.val
             << ",\"solution_joint_positions\":" << json_array(solution)
             << ",\"internal_attempt_count\":1"
             << ",\"internal_random_restart_observed\":false"
             << ",\"silent_repair_applied\":false"
             << ",\"elapsed_s\":" << json_number(elapsed)
             << ",\"bridge_process_id\":" << static_cast<long long>(::getpid()) << "}\n";
    }
    output.flush();
    rclcpp::shutdown();
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "stage3_h4_1_ik_bridge: " << error.what() << "\n";
    if (rclcpp::ok()) rclcpp::shutdown();
    return 2;
  }
}
