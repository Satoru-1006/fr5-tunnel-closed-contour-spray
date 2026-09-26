#include <algorithm>
#include <Eigen/Geometry>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <srdfdom/srdfdom/model.h>
#include <urdf/model.h>

struct Case { std::string id; std::string path; };

static std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream stream(line);
  std::string item;
  while (std::getline(stream, item, ',')) out.push_back(item);
  return out;
}

static std::vector<Case> read_cases(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open case manifest: " + path);
  std::string line;
  std::getline(input, line);
  std::vector<Case> cases;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split(line);
    if (fields.size() < 2) throw std::runtime_error("malformed case manifest row");
    cases.push_back({fields[0], fields[1]});
  }
  if (cases.empty()) throw std::runtime_error("empty case manifest");
  return cases;
}

static std::vector<std::vector<double>> read_joints(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open trajectory: " + path);
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty trajectory");
  const auto header = split(line);
  std::vector<std::size_t> q_indices;
  for (int joint = 1; joint <= 6; ++joint) {
    const std::string wanted = "j" + std::to_string(joint) + "_q";
    const auto found = std::find(header.begin(), header.end(), wanted);
    if (found == header.end()) throw std::runtime_error("trajectory missing named position column " + wanted + ": " + path);
    q_indices.push_back(static_cast<std::size_t>(std::distance(header.begin(), found)));
  }
  std::vector<std::vector<double>> rows;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split(line);
    if (fields.size() <= *std::max_element(q_indices.begin(), q_indices.end()))
      throw std::runtime_error("trajectory row is missing a named position field: " + path);
    std::vector<double> values;
    for (const auto index : q_indices) values.push_back(std::stod(fields[index]));
    rows.push_back(std::move(values));
  }
  if (rows.empty()) throw std::runtime_error("trajectory has no states");
  return rows;
}

static double number(double value) {
  return value;
}

int main(int argc, char** argv) {
  try {
    std::string cases_path, urdf_path, srdf_path, output_path;
    for (int i = 1; i + 1 < argc; i += 2) {
      const std::string key(argv[i]);
      const std::string value(argv[i + 1]);
      if (key == "--cases") cases_path = value;
      else if (key == "--urdf") urdf_path = value;
      else if (key == "--srdf") srdf_path = value;
      else if (key == "--output") output_path = value;
    }
    if (cases_path.empty() || urdf_path.empty() || srdf_path.empty() || output_path.empty())
      throw std::runtime_error("missing --cases/--urdf/--srdf/--output");

    auto urdf = std::make_shared<urdf::Model>();
    if (!urdf->initFile(urdf_path)) throw std::runtime_error("URDF parse failed");
    auto srdf = std::make_shared<srdf::Model>();
    if (!srdf->initFile(*urdf, srdf_path)) throw std::runtime_error("SRDF parse failed");
    auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
    const auto* group = model->getJointModelGroup("fairino5_v6_group");
    const auto* tip = model->getLinkModel("spray_tcp_link");
    if (!group || !tip) throw std::runtime_error("MoveIt group or TCP link missing");

    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("cannot open FK output: " + output_path);
    output << "case_id,waypoint,x_m,y_m,z_m,qx,qy,qz,qw\n";
    output << std::setprecision(17);
    rclcpp::init(argc, argv);
    for (const auto& item : read_cases(cases_path)) {
      const auto rows = read_joints(item.path);
      for (std::size_t index = 0; index < rows.size(); ++index) {
        moveit::core::RobotState state(model);
        state.setJointGroupPositions(group, rows[index]);
        state.update();
        const Eigen::Isometry3d transform = state.getGlobalLinkTransform(tip);
        const Eigen::Quaterniond orientation(transform.rotation());
        output << '"' << item.id << '"' << ',' << index << ','
               << number(transform.translation().x()) << ','
               << number(transform.translation().y()) << ','
               << number(transform.translation().z()) << ','
               << number(orientation.x()) << ',' << number(orientation.y()) << ','
               << number(orientation.z()) << ',' << number(orientation.w()) << '\n';
      }
    }
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    if (rclcpp::ok()) rclcpp::shutdown();
    std::cerr << "stage4a_fk: " << error.what() << '\n';
    return 2;
  }
}
