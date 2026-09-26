// D56 direct jerk-limited reconstruction.
//
// MoveIt's Ruckig adapter changes waypoint timing but leaves the stored
// waypoint derivatives in place.  This shadow tool uses the same active
// limits and reconstructs each local profile, emitting the analytic
// q/dq/ddq/jerk states that were actually generated.  It never writes a
// protected Stage 3/D47 artifact.

#include <ruckig/ruckig.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kDofs = 6;
struct Point { double t{}; std::array<double, kDofs> q{}, v{}, a{}; };
struct State { double t{}; std::array<double, kDofs> q{}, v{}, a{}, j{}; };

std::vector<double> split(const std::string& line) {
  std::vector<double> out; std::stringstream ss(line); std::string field;
  while (std::getline(ss, field, ',')) out.push_back(std::stod(field));
  return out;
}

std::vector<Point> read_csv(const std::string& path) {
  std::ifstream in(path); if (!in) throw std::runtime_error("cannot open input: " + path);
  std::string line; if (!std::getline(in, line)) throw std::runtime_error("empty input");
  std::vector<Point> result;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto row = split(line); if (row.size() < 19) throw std::runtime_error("trajectory needs t+18 state columns");
    Point point; point.t = row[0];
    for (std::size_t j = 0; j < kDofs; ++j) {
      point.q[j] = row[1 + j]; point.v[j] = row[7 + j]; point.a[j] = row[13 + j];
    }
    result.push_back(point);
  }
  if (result.size() < 2) throw std::runtime_error("trajectory needs two states");
  return result;
}

bool success(ruckig::Result result) {
  return result == ruckig::Result::Working || result == ruckig::Result::Finished;
}

const char* result_name(ruckig::Result result) {
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
  return "Unknown";
}

double number(double value) {
  // Preserve non-finite values in the diagnostic CSV.  Mapping NaN/Inf to
  // zero would turn an invalid native state into an apparently valid one and
  // violate the D56 no-silent-workaround rule.
  return value;
}

std::array<double, kDofs> project_velocity(const std::array<double, kDofs>& value,
                                           const std::array<double, kDofs>& limit) {
  std::array<double, kDofs> result{};
  for (std::size_t j = 0; j < kDofs; ++j) result[j] = std::clamp(value[j], -limit[j], limit[j]);
  return result;
}

std::array<double, kDofs> project_acceleration(const std::array<double, kDofs>& velocity,
                                               const std::array<double, kDofs>& acceleration,
                                               const std::array<double, kDofs>& max_velocity,
                                               const std::array<double, kDofs>& max_acceleration,
                                               const std::array<double, kDofs>& max_jerk) {
  std::array<double, kDofs> result{};
  for (std::size_t j = 0; j < kDofs; ++j) {
    double a = std::clamp(acceleration[j], -max_acceleration[j], max_acceleration[j]);
    const double v = velocity[j];
    if (a > 0.0 && v - a * a / (2.0 * max_jerk[j]) < -max_velocity[j]) {
      a = std::min(a, std::sqrt(std::max(0.0, 2.0 * max_jerk[j] * (v + max_velocity[j]))));
    }
    if (a < 0.0 && v + a * a / (2.0 * max_jerk[j]) > max_velocity[j]) {
      a = std::max(a, -std::sqrt(std::max(0.0, 2.0 * max_jerk[j] * (max_velocity[j] - v))));
    }
    result[j] = a;
  }
  return result;
}

std::array<double, kDofs> jerk_at(const ruckig::Trajectory<ruckig::DynamicDOFs, ruckig::StandardVector>& trajectory,
                                  double time, std::size_t joint) {
  const auto profile = trajectory.get_profiles()[0][joint];
  std::size_t index = 0;
  while (index + 1 < profile.t_sum.size() && time > profile.t_sum[index] + 1.0e-12) ++index;
  std::array<double, kDofs> result{};
  result[joint] = profile.j[index];
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6) {
    std::cerr << "usage: d56_ruckig_reconstruct input.csv output.csv summary.json velocity_scaling [sample_period_s]\n";
    return 2;
  }
  try {
    const auto input_points = read_csv(argv[1]);
    const double scaling = std::stod(argv[4]);
    const double sample_period = argc == 6 ? std::stod(argv[5]) : 0.0;
    if (!std::isfinite(sample_period) || sample_period < 0.0) throw std::runtime_error("sample_period_s must be finite and non-negative");
    const std::array<double, kDofs> max_velocity{3.15 * scaling, 3.15 * scaling, 3.15 * scaling, 3.2 * scaling, 3.2 * scaling, 3.2 * scaling};
    const std::array<double, kDofs> max_acceleration{0.7 * scaling, 0.7 * scaling, 0.7 * scaling, 0.7 * scaling, 0.7 * scaling, 0.7 * scaling};
    const std::array<double, kDofs> max_jerk{8.0, 8.0, 8.0, 8.0, 8.0, 8.0};
    ruckig::Ruckig<ruckig::DynamicDOFs> ruckig(kDofs, 0.01);
    std::vector<State> output;
    output.reserve(input_points.size());
    double current_time = 0.0;
    std::size_t successes = 0, failures = 0, total_samples = 0;
    double max_profile_jerk = 0.0, max_duration = 0.0;
    std::size_t first_failure = input_points.size();
    ruckig::Result first_result = ruckig::Result::Working;

    for (std::size_t i = 0; i + 1 < input_points.size(); ++i) {
      const auto current_v = project_velocity(input_points[i].v, max_velocity);
      const auto target_v = project_velocity(input_points[i + 1].v, max_velocity);
      const auto current_a = project_acceleration(current_v, input_points[i].a, max_velocity, max_acceleration, max_jerk);
      const auto target_a = project_acceleration(target_v, input_points[i + 1].a, max_velocity, max_acceleration, max_jerk);
      ruckig::InputParameter<ruckig::DynamicDOFs> input(kDofs);
      input.current_position = std::vector<double>(input_points[i].q.begin(), input_points[i].q.end());
      input.target_position = std::vector<double>(input_points[i + 1].q.begin(), input_points[i + 1].q.end());
      input.current_velocity = std::vector<double>(current_v.begin(), current_v.end());
      input.target_velocity = std::vector<double>(target_v.begin(), target_v.end());
      input.current_acceleration = std::vector<double>(current_a.begin(), current_a.end());
      input.target_acceleration = std::vector<double>(target_a.begin(), target_a.end());
      input.max_velocity = std::vector<double>(max_velocity.begin(), max_velocity.end());
      input.max_acceleration = std::vector<double>(max_acceleration.begin(), max_acceleration.end());
      input.max_jerk = std::vector<double>(max_jerk.begin(), max_jerk.end());
      ruckig::Trajectory<ruckig::DynamicDOFs, ruckig::StandardVector> trajectory(kDofs);
      const auto result = ruckig.calculate(input, trajectory);
      if (!success(result)) {
        ++failures;
        if (first_failure == input_points.size()) { first_failure = i; first_result = result; }
        continue;
      }
      ++successes;
      const double duration = trajectory.get_duration();
      max_duration = std::max(max_duration, duration);
      const auto profiles = trajectory.get_profiles();
      for (std::size_t j = 0; j < kDofs; ++j)
        for (double jerk : profiles[0][j].j) max_profile_jerk = std::max(max_profile_jerk, std::abs(jerk));
      const std::size_t samples = sample_period > 0.0
          ? std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(duration / sample_period)))
          : 1;
      std::size_t first_sample = 0;
      if (output.empty()) {
        State state; state.t = current_time; state.q = input_points[i].q; state.v = current_v; state.a = current_a;
        for (std::size_t j = 0; j < kDofs; ++j) state.j[j] = profiles[0][j].j[0];
        output.push_back(state); ++total_samples;
      } else {
        first_sample = 1;
      }
      if (sample_period == 0.0) first_sample = samples;
      else if (!output.empty()) first_sample = 1;
      for (std::size_t sample = first_sample; sample <= samples; ++sample) {
        const double local = (sample == samples) ? duration : std::min(duration, sample_period * static_cast<double>(sample));
        std::vector<double> q(kDofs), v(kDofs), a(kDofs); trajectory.at_time(local, q, v, a);
        State state; state.t = current_time + local;
        for (std::size_t j = 0; j < kDofs; ++j) {
          state.q[j] = q[j]; state.v[j] = v[j]; state.a[j] = a[j];
          state.j[j] = jerk_at(trajectory, local, j)[j];
        }
        output.push_back(state); ++total_samples;
      }
      current_time += duration;
    }
    if (failures != 0 || output.empty()) throw std::runtime_error("reconstruction_failure");
    std::ofstream csv(argv[2]); if (!csv) throw std::runtime_error("cannot open output csv");
    csv << "t"; for (std::size_t j = 1; j <= kDofs; ++j) csv << ",j" << j << "_q"; for (std::size_t j = 1; j <= kDofs; ++j) csv << ",j" << j << "_dq"; for (std::size_t j = 1; j <= kDofs; ++j) csv << ",j" << j << "_ddq"; for (std::size_t j = 1; j <= kDofs; ++j) csv << ",j" << j << "_jerk"; csv << '\n';
    csv << std::setprecision(17);
    for (const auto& state : output) {
      csv << state.t; for (double value : state.q) csv << ',' << number(value); for (double value : state.v) csv << ',' << number(value); for (double value : state.a) csv << ',' << number(value); for (double value : state.j) csv << ',' << number(value); csv << '\n';
    }
    std::ofstream summary(argv[3]); if (!summary) throw std::runtime_error("cannot open summary");
    summary << std::setprecision(17) << "{\n"
            << "  \"schema_version\":\"d56-direct-ruckig-reconstruction-v1\",\n"
            << "  \"input_state_count\":" << input_points.size() << ",\n"
            << "  \"output_state_count\":" << output.size() << ",\n"
            << "  \"segment_count\":" << input_points.size() - 1 << ",\n"
            << "  \"successful_segment_count\":" << successes << ",\n"
            << "  \"failed_segment_count\":" << failures << ",\n"
            << "  \"velocity_scaling\":" << scaling << ",\n"
            << "  \"sample_period_s\":" << sample_period << ",\n"
            << "  \"active_velocity_limits_rad_s\":[" << max_velocity[0] << "," << max_velocity[1] << "," << max_velocity[2] << "," << max_velocity[3] << "," << max_velocity[4] << "," << max_velocity[5] << "],\n"
            << "  \"active_acceleration_limits_rad_s2\":[" << max_acceleration[0] << "," << max_acceleration[1] << "," << max_acceleration[2] << "," << max_acceleration[3] << "," << max_acceleration[4] << "," << max_acceleration[5] << "],\n"
            << "  \"active_jerk_limits_rad_s3\":[8,8,8,8,8,8],\n"
            << "  \"first_failure_segment_index\":" << (first_failure == input_points.size() ? -1 : static_cast<long long>(first_failure)) << ",\n"
            << "  \"first_failure_result\":\"" << result_name(first_result) << "\",\n"
            << "  \"max_profile_jerk_rad_s3\":" << max_profile_jerk << ",\n"
            << "  \"max_segment_duration_s\":" << max_duration << ",\n"
            << "  \"total_duration_s\":" << current_time << ",\n"
            << "  \"status\":\"PASS\"\n}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "d56_ruckig_reconstruct: " << e.what() << '\n';
    return 3;
  }
}
