// D50 isolated analytic jerk oracle.
//
// Replays the exact q/v/a knot states exported immediately before MoveIt's
// apply_ruckig_smoothing call.  The result is the native Ruckig Profile.j
// representation, not a numerical derivative of sampled acceleration.

#include <ruckig/ruckig.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kDofs = 6;
const std::array<double, kDofs> kMaxVelocity{0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48};
const std::array<double, kDofs> kMaxAcceleration{0.105, 0.105, 0.105, 0.105, 0.105, 0.105};
const std::array<double, kDofs> kMaxJerk{8.0, 8.0, 8.0, 8.0, 8.0, 8.0};

struct Point {
  int primitive{};
  std::size_t local{};
  double time{};
  std::array<double, kDofs> q{}, v{}, a{};
};

struct Hotspot {
  std::size_t segment{};
  int primitive{};
  std::size_t local{};
  std::size_t joint{};
  std::string phase;
  std::size_t phase_index{};
  double jerk{};
  double duration{};
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
  if (!input) throw std::runtime_error("cannot open oracle input");
  std::string line;
  std::getline(input, line);
  std::vector<Point> points;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split(line);
    if (fields.size() != 21) throw std::runtime_error("oracle input must contain 21 columns");
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

void number(std::ostream& out, double value) {
  if (std::isfinite(value)) out << std::setprecision(17) << value;
  else out << "null";
}

void add_hotspot(std::vector<Hotspot>& hotspots, std::size_t segment, const Point& point,
                 std::size_t joint, const char* phase, std::size_t phase_index,
                 double jerk, double duration) {
  hotspots.push_back({segment, point.primitive, point.local, joint, phase, phase_index, jerk, duration});
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    return 2;
  }
  try {
    const auto points = read_csv(argv[1]);
    const std::string summary_path = argv[2];
    const std::string hotspot_path = argv[3];
    ruckig::Ruckig<ruckig::DynamicDOFs> ruckig(kDofs, 0.01);
    std::vector<Hotspot> hotspots;
    std::size_t segments = 0, strict_valid = 0, successful = 0;
    std::size_t invalid_input = 0, other_errors = 0;
    double max_abs_jerk = 0.0, total_duration = 0.0, max_duration = 0.0;
    int max_primitive = -1;
    std::size_t max_local = 0, max_joint = 0, max_segment = 0;
    std::string max_phase = "none";
    std::size_t max_phase_index = 0;
    ruckig::Result first_error = ruckig::Result::Working;
    int first_error_primitive = -1;
    std::size_t first_error_local = 0;

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
      const bool valid = ruckig.validate_input(input, true, true);
      strict_valid += valid;
      ruckig::Trajectory<ruckig::DynamicDOFs, ruckig::StandardVector> trajectory(kDofs);
      const auto result = ruckig.calculate(input, trajectory);
      if (!success(result)) {
        if (result == ruckig::Result::ErrorInvalidInput) ++invalid_input;
        else ++other_errors;
        if (first_error_primitive < 0) {
          first_error = result;
          first_error_primitive = current.primitive;
          first_error_local = current.local;
        }
        continue;
      }
      ++successful;
      total_duration += trajectory.get_duration();
      max_duration = std::max(max_duration, trajectory.get_duration());
      const auto profiles = trajectory.get_profiles();
      for (std::size_t joint = 0; joint < kDofs; ++joint) {
        const auto& profile = profiles[0][joint];
        for (std::size_t phase = 0; phase < profile.j.size(); ++phase) {
          add_hotspot(hotspots, index, current, joint, "profile", phase, profile.j[phase], profile.t[phase]);
          const double magnitude = std::abs(profile.j[phase]);
          if (magnitude > max_abs_jerk) {
            max_abs_jerk = magnitude;
            max_primitive = current.primitive;
            max_local = current.local;
            max_joint = joint;
            max_segment = index;
            max_phase = "profile";
            max_phase_index = phase;
          }
        }
        for (std::size_t phase = 0; phase < profile.brake.j.size(); ++phase) {
          if (profile.brake.t[phase] <= 0.0) continue;
          add_hotspot(hotspots, index, current, joint, "brake", phase, profile.brake.j[phase], profile.brake.t[phase]);
          const double magnitude = std::abs(profile.brake.j[phase]);
          if (magnitude > max_abs_jerk) {
            max_abs_jerk = magnitude;
            max_primitive = current.primitive;
            max_local = current.local;
            max_joint = joint;
            max_segment = index;
            max_phase = "brake";
            max_phase_index = phase;
          }
        }
      }
    }

    std::sort(hotspots.begin(), hotspots.end(), [](const Hotspot& left, const Hotspot& right) {
      return std::abs(left.jerk) > std::abs(right.jerk);
    });
    std::ofstream hotspot_out(hotspot_path, std::ios::trunc);
    if (!hotspot_out) throw std::runtime_error("cannot open hotspot output");
    hotspot_out << std::setprecision(17);
    for (const auto& item : hotspots) {
      hotspot_out << "{\"segment\":" << item.segment << ",\"primitive\":" << item.primitive
                  << ",\"local\":" << item.local << ",\"joint\":" << item.joint
                  << ",\"phase\":\"" << item.phase << "\",\"phase_index\":" << item.phase_index
                  << ",\"jerk_rad_s3\":" << item.jerk << ",\"duration_s\":" << item.duration << "}\n";
    }

    std::ofstream out(summary_path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open summary output");
    out << "{\n  \"schema_version\": \"d50-native-ruckig-profile-oracle-v1\",\n"
        << "  \"input_state_count\": " << points.size() << ",\n"
        << "  \"segments\": " << segments << ",\n"
        << "  \"strict_current_target_valid\": " << strict_valid << ",\n"
        << "  \"successful_native_profiles\": " << successful << ",\n"
        << "  \"error_invalid_input_count\": " << invalid_input << ",\n"
        << "  \"other_error_count\": " << other_errors << ",\n"
        << "  \"max_abs_analytic_jerk_rad_s3\": "; number(out, max_abs_jerk); out << ",\n"
        << "  \"max_analytic_jerk_ratio\": "; number(out, max_abs_jerk / 8.0); out << ",\n"
        << "  \"max_primitive_index\": " << max_primitive << ",\n"
        << "  \"max_local_waypoint\": " << max_local << ",\n"
        << "  \"max_segment\": " << max_segment << ",\n"
        << "  \"max_joint_index\": " << max_joint << ",\n"
        << "  \"max_phase\": \"" << max_phase << "\",\n"
        << "  \"max_phase_index\": " << max_phase_index << ",\n"
        << "  \"total_native_duration_s\": "; number(out, total_duration); out << ",\n"
        << "  \"maximum_native_segment_duration_s\": "; number(out, max_duration); out << ",\n"
        << "  \"first_error_primitive_index\": " << first_error_primitive << ",\n"
        << "  \"first_error_local_waypoint\": " << first_error_local << ",\n"
        << "  \"first_error_result\": \"" << result_name(first_error) << "\",\n"
        << "  \"jerk_truth\": \"NATIVE_RUCKIG_PROFILE_J\",\n"
        << "  \"passed\": " << ((segments > 0 && successful == segments && strict_valid == segments && invalid_input == 0 && other_errors == 0 && max_abs_jerk <= 8.0 + 1.0e-9) ? "true" : "false") << "\n}\n";
    return 0;
  } catch (const std::exception&) {
    return 3;
  }
}
