// Stage 3 H7.7 lightweight native Ruckig oracle.
//
// This is an offline, read-only candidate evaluator. It reconstructs every
// adjacent Ruckig call from an exported q/v/a knot CSV, performs strict native
// input validation, calculates the exact native profile, and applies MoveIt's
// frozen 0.01 s overshoot sampling semantics. It has no ROS, action, controller,
// ledger, or robot-execution surface.

#include <ruckig/ruckig.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kDofs = 6;
constexpr double kOvershootPeriod = 0.01;
const std::array<double, kDofs> kMaxVelocity{0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48};
const std::array<double, kDofs> kMaxAcceleration{0.105, 0.105, 0.105, 0.105, 0.105, 0.105};
const std::array<double, kDofs> kMaxJerk{8.0, 8.0, 8.0, 8.0, 8.0, 8.0};

struct Point {
  int primitive{};
  std::size_t local{};
  double time{};
  std::array<double, kDofs> q{}, v{}, a{};
};

struct Overshoot {
  bool positive{false};
  std::size_t joint{};
  double time{};
  double magnitude{};
};

std::vector<double> split(const std::string& line) {
  std::vector<double> values;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) values.push_back(std::stod(field));
  return values;
}

std::vector<Point> read_csv(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open candidate CSV");
  std::string line;
  std::getline(input, line);
  std::vector<Point> points;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split(line);
    if (fields.size() != 21) throw std::runtime_error("candidate CSV must contain 21 columns");
    Point point;
    point.primitive = static_cast<int>(fields[0]);
    point.local = static_cast<std::size_t>(fields[1]);
    point.time = fields[2];
    for (std::size_t j = 0; j < kDofs; ++j) {
      point.q[j] = fields[3 + j];
      point.v[j] = fields[9 + j];
      point.a[j] = fields[15 + j];
    }
    points.push_back(point);
  }
  return points;
}

bool success(const ruckig::Result result) {
  return result == ruckig::Result::Working || result == ruckig::Result::Finished;
}

const char* result_name(const ruckig::Result result) {
  switch (result) {
    case ruckig::Result::Working: return "Working";
    case ruckig::Result::Finished: return "Finished";
    case ruckig::Result::Error: return "Error";
    case ruckig::Result::ErrorInvalidInput: return "ErrorInvalidInput";
    case ruckig::Result::ErrorTrajectoryDuration: return "ErrorTrajectoryDuration";
    case ruckig::Result::ErrorPositionalLimits: return "ErrorPositionalLimits";
    case ruckig::Result::ErrorExecutionTimeCalculation: return "ErrorExecutionTimeCalculation";
    case ruckig::Result::ErrorSynchronizationCalculation: return "ErrorSynchronizationCalculation";
  }
  return "unknown";
}

void number(std::ostream& out, const double value) {
  if (std::isfinite(value)) out << std::setprecision(17) << value;
  else out << "null";
}

Overshoot check_overshoot(
    ruckig::Trajectory<ruckig::DynamicDOFs, ruckig::StandardVector>& trajectory,
    const ruckig::InputParameter<ruckig::DynamicDOFs>& input,
    const double threshold) {
  Overshoot worst;
  for (double time = kOvershootPeriod; time < trajectory.get_duration(); time += kOvershootPeriod) {
    std::vector<double> q(kDofs), v(kDofs), a(kDofs);
    std::size_t section = 0;
    trajectory.at_time(time, q, v, a, section);
    (void)section;
    for (std::size_t joint = 0; joint < kDofs; ++joint) {
      const double denominator = input.current_position[joint] - input.target_position[joint];
      if (denominator == 0.0) continue;
      const double residual = q[joint] - input.target_position[joint];
      const double magnitude = std::abs(residual);
      if (residual / denominator < 0.0 && magnitude > threshold && magnitude > worst.magnitude) {
        worst = {true, joint, time, magnitude};
      }
    }
  }
  return worst;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: stage3_h7_7_ruckig_oracle <candidate.csv> <summary.json> <threshold>\n";
    return 2;
  }
  try {
    const auto points = read_csv(argv[1]);
    const double threshold = std::stod(argv[3]);
    ruckig::Ruckig<ruckig::DynamicDOFs> validator(kDofs, kOvershootPeriod);
    std::size_t segments = 0, current_valid = 0, target_valid = 0, strict_valid = 0;
    std::size_t invalid_input = 0, other_errors = 0, overshoot_segments = 0;
    double worst_magnitude = 0.0, total_duration = 0.0, max_duration = 0.0;
    int worst_primitive = -1;
    std::size_t worst_local = 0, worst_joint = 0;
    double worst_time = 0.0;
    int first_error_primitive = -1;
    std::size_t first_error_local = 0;
    ruckig::Result first_error = ruckig::Result::Working;

    for (std::size_t index = 0; index + 1 < points.size(); ++index) {
      const auto& current = points[index];
      const auto& target = points[index + 1];
      if (current.primitive != target.primitive) continue;
      ++segments;
      ruckig::InputParameter<ruckig::DynamicDOFs> input(kDofs);
      input.current_position = std::vector<double>(current.q.begin(), current.q.end());
      input.target_position = std::vector<double>(target.q.begin(), target.q.end());
      input.current_velocity = std::vector<double>(current.v.begin(), current.v.end());
      input.target_velocity = std::vector<double>(target.v.begin(), target.v.end());
      input.current_acceleration = std::vector<double>(current.a.begin(), current.a.end());
      input.target_acceleration = std::vector<double>(target.a.begin(), target.a.end());
      input.max_velocity = std::vector<double>(kMaxVelocity.begin(), kMaxVelocity.end());
      input.max_acceleration = std::vector<double>(kMaxAcceleration.begin(), kMaxAcceleration.end());
      input.max_jerk = std::vector<double>(kMaxJerk.begin(), kMaxJerk.end());

      const bool cv = validator.validate_input(input, true, false);
      const bool tv = validator.validate_input(input, false, true);
      const bool sv = validator.validate_input(input, true, true);
      current_valid += cv;
      target_valid += tv;
      strict_valid += sv;
      ruckig::Trajectory<ruckig::DynamicDOFs, ruckig::StandardVector> trajectory(kDofs);
      const auto result = validator.calculate(input, trajectory);
      if (!success(result)) {
        if (result == ruckig::Result::ErrorInvalidInput) ++invalid_input;
        else ++other_errors;
        if (first_error_primitive < 0) {
          first_error_primitive = current.primitive;
          first_error_local = current.local;
          first_error = result;
        }
        continue;
      }
      total_duration += trajectory.get_duration();
      max_duration = std::max(max_duration, trajectory.get_duration());
      const auto overshoot = check_overshoot(trajectory, input, threshold);
      if (overshoot.positive) {
        ++overshoot_segments;
        if (overshoot.magnitude > worst_magnitude) {
          worst_magnitude = overshoot.magnitude;
          worst_primitive = current.primitive;
          worst_local = current.local;
          worst_joint = overshoot.joint;
          worst_time = overshoot.time;
        }
      }
    }

    std::ofstream out(argv[2], std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open summary JSON");
    out << "{\n  \"schema_version\": \"stage3-h7-7-native-oracle-v1\",\n";
    out << "  \"segments\": " << segments << ",\n";
    out << "  \"strict_current_valid\": " << current_valid << ",\n";
    out << "  \"strict_target_valid\": " << target_valid << ",\n";
    out << "  \"strict_current_target_valid\": " << strict_valid << ",\n";
    out << "  \"error_invalid_input_count\": " << invalid_input << ",\n";
    out << "  \"other_error_count\": " << other_errors << ",\n";
    out << "  \"overshoot_segment_count\": " << overshoot_segments << ",\n";
    out << "  \"max_joint_overshoot_rad\": "; number(out, worst_magnitude); out << ",\n";
    out << "  \"worst_primitive_index\": " << worst_primitive << ",\n";
    out << "  \"worst_local_waypoint\": " << worst_local << ",\n";
    out << "  \"worst_joint_index\": " << worst_joint << ",\n";
    out << "  \"worst_local_time_s\": "; number(out, worst_time); out << ",\n";
    out << "  \"total_native_duration_s\": "; number(out, total_duration); out << ",\n";
    out << "  \"maximum_native_segment_duration_s\": "; number(out, max_duration); out << ",\n";
    out << "  \"first_error_primitive_index\": " << first_error_primitive << ",\n";
    out << "  \"first_error_local_waypoint\": " << first_error_local << ",\n";
    out << "  \"first_error_result\": \"" << result_name(first_error) << "\",\n";
    out << "  \"passed\": " << ((strict_valid == segments && invalid_input == 0 && other_errors == 0 && overshoot_segments == 0) ? "true" : "false") << "\n}\n";
    out.flush();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 3;
  }
}
