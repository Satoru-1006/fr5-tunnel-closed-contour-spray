// D58 shadow-only analytic jerk oracle.
//
// Replays the exact q/v/a knot states captured immediately before MoveIt's
// native Ruckig smoothing.  The reported jerk is Ruckig Profile.j, not a
// finite difference of sampled acceleration.  Limits are explicit arguments
// so the oracle matches the native scaling used by the candidate run.

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

struct Point {
  double time{};
  std::array<double, kDofs> q{}, v{}, a{};
};

std::vector<double> split(const std::string& line) {
  std::vector<double> values;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) values.push_back(std::stod(field));
  return values;
}

std::array<double, kDofs> read_limit(const std::string& text) {
  std::stringstream stream(text);
  std::string field;
  std::vector<double> values;
  while (std::getline(stream, field, ',')) values.push_back(std::stod(field));
  if (values.size() == 1) return {values[0], values[0], values[0], values[0], values[0], values[0]};
  if (values.size() != kDofs) throw std::runtime_error("limit must be scalar or six comma-separated values");
  std::array<double, kDofs> result{};
  std::copy(values.begin(), values.end(), result.begin());
  return result;
}

std::vector<Point> read_csv(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open oracle input: " + path);
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty oracle input");
  std::vector<Point> points;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split(line);
    if (fields.size() != 21 && fields.size() < 25)
      throw std::runtime_error("oracle input must contain a 21-column TOTG or 25-column native trajectory");
    Point point;
    const std::size_t q_offset = fields.size() == 21 ? 3 : 1;
    const std::size_t v_offset = fields.size() == 21 ? 9 : 7;
    const std::size_t a_offset = fields.size() == 21 ? 15 : 13;
    point.time = fields[fields.size() == 21 ? 2 : 0];
    for (std::size_t joint = 0; joint < kDofs; ++joint) {
      point.q[joint] = fields[q_offset + joint];
      point.v[joint] = fields[v_offset + joint];
      point.a[joint] = fields[a_offset + joint];
    }
    points.push_back(point);
  }
  if (points.size() < 2) throw std::runtime_error("oracle input has fewer than two states");
  return points;
}

bool success(const ruckig::Result result) {
  return result == ruckig::Result::Working || result == ruckig::Result::Finished;
}

bool overshoots(ruckig::Trajectory<ruckig::DynamicDOFs, ruckig::StandardVector>& trajectory,
                const ruckig::InputParameter<ruckig::DynamicDOFs>& input, double threshold) {
  for (double time = 0.01; time < trajectory.get_duration(); time += 0.01) {
    std::vector<double> q(kDofs), velocity(kDofs), acceleration(kDofs);
    trajectory.at_time(time, q, velocity, acceleration);
    for (std::size_t joint = 0; joint < kDofs; ++joint) {
      const double denominator = input.current_position[joint] - input.target_position[joint];
      if (denominator == 0.0) continue;
      const double residual = q[joint] - input.target_position[joint];
      if (residual / denominator < 0.0 && std::abs(residual) > threshold) return true;
    }
  }
  return false;
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

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7 && argc != 8) {
    std::cerr << "usage: d58_ruckig_profile_oracle <knots.csv> <summary.json> <hotspots.jsonl> <max_velocity> <max_acceleration> <max_jerk> [overshoot_threshold]\n";
    return 2;
  }
  try {
    const auto points = read_csv(argv[1]);
    const auto velocity_limit = read_limit(argv[4]);
    const auto acceleration_limit = read_limit(argv[5]);
    const auto jerk_limit = read_limit(argv[6]);
    const double overshoot_threshold = argc == 8 ? std::stod(argv[7]) : 0.005;
    if (!(std::isfinite(overshoot_threshold) && overshoot_threshold >= 0.0))
      throw std::runtime_error("overshoot threshold must be finite and non-negative");
    for (std::size_t joint = 0; joint < kDofs; ++joint) {
      if (!(std::isfinite(velocity_limit[joint]) && velocity_limit[joint] > 0.0 &&
            std::isfinite(acceleration_limit[joint]) && acceleration_limit[joint] > 0.0 &&
            std::isfinite(jerk_limit[joint]) && jerk_limit[joint] > 0.0))
        throw std::runtime_error("limits must be finite and positive");
    }
    double average_dt = 0.01;
    if (points.size() > 1) {
      average_dt = (points.back().time - points.front().time) / static_cast<double>(points.size() - 1);
      if (!std::isfinite(average_dt) || average_dt <= 0.0) average_dt = 0.01;
    }
    ruckig::Ruckig<ruckig::DynamicDOFs> ruckig(kDofs, average_dt);
    std::size_t strict_valid = 0;
    std::size_t successful = 0;
    std::size_t invalid_input = 0;
    std::size_t other_errors = 0;
    std::size_t overshoot_segments = 0;
    std::size_t extended_segments = 0;
    double maximum_extension_factor = 1.0;
    double max_abs_jerk = 0.0;
    std::size_t max_segment = 0;
    std::size_t max_joint = 0;
    std::string first_error = "Working";
    std::ofstream hotspots(argv[3], std::ios::trunc);
    if (!hotspots) throw std::runtime_error("cannot open hotspot output");
    hotspots << std::setprecision(17);
    for (std::size_t index = 0; index + 1 < points.size(); ++index) {
      const auto& current = points[index];
      const auto& target = points[index + 1];
      const double original_dt = target.time - current.time;
      if (!std::isfinite(original_dt) || original_dt <= 0.0) throw std::runtime_error("non-increasing knot time");
      double extension_factor = 1.0;
      bool solved = false;
      bool had_extension = false;
      ruckig::Trajectory<ruckig::DynamicDOFs, ruckig::StandardVector> trajectory(kDofs);
      ruckig::InputParameter<ruckig::DynamicDOFs> input(kDofs);
      ruckig::Result result = ruckig::Result::Error;
      for (;;) {
        input.current_position = std::vector<double>(current.q.begin(), current.q.end());
        input.target_position = std::vector<double>(target.q.begin(), target.q.end());
        input.current_velocity = std::vector<double>(current.v.begin(), current.v.end());
        input.target_velocity = std::vector<double>(target.v.begin(), target.v.end());
        input.current_acceleration = std::vector<double>(current.a.begin(), current.a.end());
        input.target_acceleration = std::vector<double>(target.a.begin(), target.a.end());
        input.max_velocity = std::vector<double>(velocity_limit.begin(), velocity_limit.end());
        input.max_acceleration = std::vector<double>(acceleration_limit.begin(), acceleration_limit.end());
        input.max_jerk = std::vector<double>(jerk_limit.begin(), jerk_limit.end());
        if (extension_factor > 1.0) {
          for (std::size_t joint = 0; joint < kDofs; ++joint) {
            input.target_velocity[joint] = target.v[joint] / extension_factor;
            input.target_acceleration[joint] =
                (input.target_velocity[joint] - current.v[joint]) / (original_dt * extension_factor);
          }
        }
        // Match MoveIt's getNextRuckigInput(): clamp at every boundary.
        for (std::size_t joint = 0; joint < kDofs; ++joint) {
          input.current_velocity[joint] = std::clamp(input.current_velocity[joint], -velocity_limit[joint], velocity_limit[joint]);
          input.target_velocity[joint] = std::clamp(input.target_velocity[joint], -velocity_limit[joint], velocity_limit[joint]);
          input.current_acceleration[joint] = std::clamp(input.current_acceleration[joint], -acceleration_limit[joint], acceleration_limit[joint]);
          input.target_acceleration[joint] = std::clamp(input.target_acceleration[joint], -acceleration_limit[joint], acceleration_limit[joint]);
        }
        const bool valid = ruckig.validate_input(input, true, true);
        strict_valid += valid ? 1 : 0;
        result = ruckig.calculate(input, trajectory);
        const bool overshoot = success(result) && overshoots(trajectory, input, overshoot_threshold);
        if (success(result) && !overshoot) {
          solved = true;
          break;
        }
        if (overshoot) ++overshoot_segments;
        if (!success(result)) {
          if (result == ruckig::Result::ErrorInvalidInput) ++invalid_input;
          else ++other_errors;
          if (first_error == "Working") first_error = result_name(result);
        }
        extension_factor *= 1.1;
        had_extension = true;
        if (extension_factor > 50.0) break;
      }
      if (!solved) continue;
      ++successful;
      if (had_extension) ++extended_segments;
      maximum_extension_factor = std::max(maximum_extension_factor, extension_factor);
      const auto profiles = trajectory.get_profiles();
      for (std::size_t joint = 0; joint < kDofs; ++joint) {
        const auto& profile = profiles[0][joint];
        for (std::size_t phase = 0; phase < profile.j.size(); ++phase) {
          const double value = profile.j[phase];
          hotspots << "{\"segment\":" << index << ",\"joint\":" << joint << ",\"phase\":" << phase << ",\"jerk_rad_s3\":" << value << ",\"duration_s\":" << profile.t[phase] << "}\n";
          if (std::abs(value) > max_abs_jerk) { max_abs_jerk = std::abs(value); max_segment = index; max_joint = joint; }
        }
        for (std::size_t phase = 0; phase < profile.brake.j.size(); ++phase) {
          if (profile.brake.t[phase] <= 0.0) continue;
          const double value = profile.brake.j[phase];
          hotspots << "{\"segment\":" << index << ",\"joint\":" << joint << ",\"phase\":" << (phase + profile.j.size()) << ",\"jerk_rad_s3\":" << value << ",\"duration_s\":" << profile.brake.t[phase] << "}\n";
          if (std::abs(value) > max_abs_jerk) { max_abs_jerk = std::abs(value); max_segment = index; max_joint = joint; }
        }
      }
    }
    std::ofstream summary(argv[2], std::ios::trunc);
    if (!summary) throw std::runtime_error("cannot open summary output");
    summary << "{\n"
            << "  \"schema_version\": \"d58-native-ruckig-profile-oracle-v1\",\n"
            << "  \"input_state_count\": " << points.size() << ",\n"
            << "  \"segments\": " << (points.size() - 1) << ",\n"
            << "  \"strict_current_target_valid\": " << strict_valid << ",\n"
            << "  \"successful_native_profiles\": " << successful << ",\n"
            << "  \"error_invalid_input_count\": " << invalid_input << ",\n"
            << "  \"other_error_count\": " << other_errors << ",\n"
            << "  \"overshoot_segment_count\": " << overshoot_segments << ",\n"
            << "  \"extended_segment_count\": " << extended_segments << ",\n"
            << "  \"maximum_extension_factor\": "; number(summary, maximum_extension_factor); summary << ",\n"
            << "  \"max_abs_analytic_jerk_rad_s3\": "; number(summary, max_abs_jerk); summary << ",\n"
            << "  \"max_analytic_jerk_ratio\": "; number(summary, max_abs_jerk / *std::max_element(jerk_limit.begin(), jerk_limit.end())); summary << ",\n"
            << "  \"max_segment\": " << max_segment << ",\n"
            << "  \"max_joint_index\": " << max_joint << ",\n"
            << "  \"first_error_result\": \"" << first_error << "\",\n"
            << "  \"overshoot_threshold\": "; number(summary, overshoot_threshold); summary << ",\n"
            << "  \"limits\": {\"max_velocity_rad_s\": [";
    for (std::size_t joint = 0; joint < kDofs; ++joint) { if (joint) summary << ","; number(summary, velocity_limit[joint]); }
    summary << "],\"max_acceleration_rad_s2\": [";
    for (std::size_t joint = 0; joint < kDofs; ++joint) { if (joint) summary << ","; number(summary, acceleration_limit[joint]); }
    summary << "],\"max_jerk_rad_s3\": [";
    for (std::size_t joint = 0; joint < kDofs; ++joint) { if (joint) summary << ","; number(summary, jerk_limit[joint]); }
    summary << "]},\n"
            << "  \"jerk_truth\": \"NATIVE_RUCKIG_PROFILE_J\",\n"
            << "  \"passed\": " << ((successful == points.size() - 1 && invalid_input == 0 && other_errors == 0 && overshoot_segments == 0 && max_abs_jerk <= *std::max_element(jerk_limit.begin(), jerk_limit.end()) + 1.0e-9) ? "true" : "false") << "\n}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 3;
  }
}
