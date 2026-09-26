// D56 evidence probe for the MoveIt Ruckig input contract.
// It mirrors MoveIt's per-segment clamping and records the first validation or
// calculation failure without modifying any project trajectory.

#include <ruckig/ruckig.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kDofs = 6;

struct Point { double t{}; std::array<double, kDofs> q{}, v{}, a{}; };

std::vector<double> split(const std::string& line) {
  std::vector<double> out;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, ',')) out.push_back(std::stod(field));
  return out;
}

std::vector<Point> read_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open input: " + path);
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("empty input");
  std::vector<Point> points;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<double> row = split(line);
    if (row.size() < 19) throw std::runtime_error("trajectory needs t+18 state columns");
    Point p;
    p.t = row[0];
    for (std::size_t j = 0; j < kDofs; ++j) {
      p.q[j] = row[1 + j];
      p.v[j] = row[7 + j];
      p.a[j] = row[13 + j];
    }
    points.push_back(p);
  }
  if (points.size() < 2) throw std::runtime_error("trajectory needs two states");
  return points;
}

bool success(ruckig::Result r) { return r == ruckig::Result::Working || r == ruckig::Result::Finished; }

const char* result_name(ruckig::Result r) {
  switch (r) {
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

std::string json_number(double v) {
  if (!std::isfinite(v)) return "null";
  std::ostringstream out; out << std::setprecision(17) << v; return out.str();
}

void print_array(std::ostream& out, const std::array<double, kDofs>& a) {
  out << '[';
  for (std::size_t i = 0; i < kDofs; ++i) { if (i) out << ','; out << json_number(a[i]); }
  out << ']';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: d56_ruckig_contract_probe trajectory.csv summary.json first_failure.json [velocity_scaling]\n";
    return 2;
  }
  try {
    const auto points = read_csv(argv[1]);
    const double velocity_scaling = argc == 5 ? std::stod(argv[4]) : 0.075;
    const std::array<double, kDofs> max_v{3.15 * velocity_scaling, 3.15 * velocity_scaling, 3.15 * velocity_scaling,
                                          3.2 * velocity_scaling, 3.2 * velocity_scaling, 3.2 * velocity_scaling};
    const std::array<double, kDofs> max_a{0.7 * velocity_scaling, 0.7 * velocity_scaling, 0.7 * velocity_scaling,
                                          0.7 * velocity_scaling, 0.7 * velocity_scaling, 0.7 * velocity_scaling};
    const std::array<double, kDofs> max_j{8.0, 8.0, 8.0, 8.0, 8.0, 8.0};
    ruckig::Ruckig<ruckig::DynamicDOFs> ruckig(kDofs, 0.01);
    std::ofstream failure(argv[3]);
    if (!failure) throw std::runtime_error("cannot open failure output");
    failure << "{\"schema_version\":\"d56-ruckig-first-failure-v1\",\"failures\":[";
    bool first = true;
    std::size_t strict_valid = 0, calculated = 0, failures = 0;
    double max_profile_jerk = 0.0;
    std::size_t first_segment = std::numeric_limits<std::size_t>::max();
    std::string first_validation_error;
    ruckig::Result first_result = ruckig::Result::Working;
    for (std::size_t i = 0; i + 1 < points.size(); ++i) {
      const auto& c = points[i]; const auto& t = points[i + 1];
      ruckig::InputParameter<ruckig::DynamicDOFs> input(kDofs);
      input.current_position = std::vector<double>(c.q.begin(), c.q.end());
      input.target_position = std::vector<double>(t.q.begin(), t.q.end());
      input.current_velocity = std::vector<double>(c.v.begin(), c.v.end());
      input.target_velocity = std::vector<double>(t.v.begin(), t.v.end());
      input.current_acceleration = std::vector<double>(c.a.begin(), c.a.end());
      input.target_acceleration = std::vector<double>(t.a.begin(), t.a.end());
      input.max_velocity = std::vector<double>(max_v.begin(), max_v.end());
      input.max_acceleration = std::vector<double>(max_a.begin(), max_a.end());
      input.max_jerk = std::vector<double>(max_j.begin(), max_j.end());
      for (std::size_t j = 0; j < kDofs; ++j) {
        input.current_velocity[j] = std::clamp(input.current_velocity[j], -max_v[j], max_v[j]);
        input.target_velocity[j] = std::clamp(input.target_velocity[j], -max_v[j], max_v[j]);
        input.current_acceleration[j] = std::clamp(input.current_acceleration[j], -max_a[j], max_a[j]);
        input.target_acceleration[j] = std::clamp(input.target_acceleration[j], -max_a[j], max_a[j]);
      }
      bool valid = false;
      try { valid = ruckig.validate_input(input, true, true); }
      catch (const std::exception& e) { first_validation_error = e.what(); }
      if (valid) ++strict_valid;
      ruckig::Trajectory<ruckig::DynamicDOFs, ruckig::StandardVector> trajectory(kDofs);
      const auto result = ruckig.calculate(input, trajectory);
      if (success(result)) {
        ++calculated;
        const auto profiles = trajectory.get_profiles();
        for (std::size_t j = 0; j < kDofs; ++j)
          for (double jerk : profiles[0][j].j) max_profile_jerk = std::max(max_profile_jerk, std::abs(jerk));
      } else {
        ++failures;
        if (first_segment == std::numeric_limits<std::size_t>::max()) {
          first_segment = i; first_result = result;
          if (first) first = false; else failure << ',';
          failure << "{\"segment_index\":" << i << ",\"time_start_s\":" << json_number(c.t)
                  << ",\"time_end_s\":" << json_number(t.t) << ",\"result\":\"" << result_name(result)
                  << "\",\"current_position\":"; print_array(failure, c.q);
          failure << ",\"current_velocity\":"; print_array(failure, c.v);
          failure << ",\"current_acceleration\":"; print_array(failure, c.a);
          failure << ",\"target_position\":"; print_array(failure, t.q);
          failure << ",\"target_velocity\":"; print_array(failure, t.v);
          failure << ",\"target_acceleration\":"; print_array(failure, t.a);
          failure << ",\"max_velocity\":"; print_array(failure, max_v);
          failure << ",\"max_acceleration\":"; print_array(failure, max_a);
          failure << ",\"max_jerk\":"; print_array(failure, max_j);
          failure << ",\"validation_error\":\"";
          for (char ch : first_validation_error) { if (ch == '\"' || ch == '\\') failure << '\\'; failure << ch; }
          failure << "\"}";
        }
      }
    }
    failure << "]}\n";
    std::ofstream summary(argv[2]);
    if (!summary) throw std::runtime_error("cannot open summary output");
    summary << "{\n  \"schema_version\":\"d56-ruckig-contract-probe-v1\",\n"
            << "  \"input_state_count\":" << points.size() << ",\n"
            << "  \"segment_count\":" << points.size() - 1 << ",\n"
            << "  \"strict_current_target_valid_count\":" << strict_valid << ",\n"
            << "  \"calculated_success_count\":" << calculated << ",\n"
            << "  \"calculation_failure_count\":" << failures << ",\n"
            << "  \"velocity_scaling\":" << json_number(velocity_scaling) << ",\n"
            << "  \"first_failure_segment_index\":" << (first_segment == std::numeric_limits<std::size_t>::max() ? -1 : static_cast<long long>(first_segment)) << ",\n"
            << "  \"first_failure_result\":\"" << result_name(first_result) << "\",\n"
            << "  \"first_validation_error\":\"";
    for (char ch : first_validation_error) { if (ch == '\"' || ch == '\\') summary << '\\'; summary << ch; }
    summary << "\",\n  \"max_analytic_profile_jerk_rad_s3\":" << json_number(max_profile_jerk) << ",\n"
            << "  \"max_jerk_limit_rad_s3\":8.0,\n"
            << "  \"active_limits\":{\"velocity_rad_s\":[" << max_v[0] << "," << max_v[1] << "," << max_v[2] << "," << max_v[3] << "," << max_v[4] << "," << max_v[5]
            << "],\"acceleration_rad_s2\":[" << max_a[0] << "," << max_a[1] << "," << max_a[2] << "," << max_a[3] << "," << max_a[4] << "," << max_a[5]
            << "],\"jerk_rad_s3\":[8,8,8,8,8,8]},\n"
            << "  \"passed\":" << ((failures == 0 && strict_valid == points.size() - 1) ? "true" : "false") << "\n}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "d56_ruckig_contract_probe: " << e.what() << '\n';
    return 3;
  }
}
