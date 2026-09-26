// Stage 2.7S probe: replay the frozen Stage 2.5R2 q/v/a knots with the
// exact distro Ruckig and explicit frozen position bounds.  This is a
// read-only probe of the immutable source; it does not produce a candidate
// trajectory or mutate any Stage 2.5R2 artifact.

#include <ruckig/ruckig.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kDofs = 6;
const std::array<double, kDofs> kMaxVelocity{0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48};
const std::array<double, kDofs> kMaxAcceleration{0.105, 0.105, 0.105, 0.105, 0.105, 0.105};
const std::array<double, kDofs> kMaxJerk{8.0, 8.0, 8.0, 8.0, 8.0, 8.0};
const std::array<double, kDofs> kMinPosition{-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543};
const std::array<double, kDofs> kMaxPosition{3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543};

struct Point { double t{}; std::array<double, kDofs> q{}, v{}, a{}; };

std::vector<double> split(const std::string& line) {
  std::vector<double> values;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) values.push_back(std::stod(field));
  return values;
}

std::vector<Point> read_csv(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open input CSV");
  std::string line;
  std::getline(input, line);
  std::vector<Point> points;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split(line);
    if (fields.size() < 19) throw std::runtime_error("malformed trajectory row");
    Point point;
    point.t = fields[0];
    for (std::size_t j = 0; j < kDofs; ++j) {
      point.q[j] = fields[1 + j];
      point.v[j] = fields[7 + j];
      point.a[j] = fields[13 + j];
    }
    points.push_back(point);
  }
  return points;
}

std::string result_name(const ruckig::Result result) {
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

void vector_json(std::ostream& out, const std::vector<double>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; number(out, values[i]); }
  out << ']';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: stage27s_bounded_ruckig_probe <trajectory.csv> <output.jsonl>\n";
    return 2;
  }
  try {
    const auto points = read_csv(argv[1]);
    if (points.size() < 2) throw std::runtime_error("trajectory has fewer than two points");
    std::ofstream output(argv[2], std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open output JSONL");

    ruckig::Ruckig<ruckig::DynamicDOFs> ruckig(kDofs, 0.01);
    std::size_t successful = 0;
    std::size_t positional_errors = 0;
    std::size_t other_errors = 0;
    int first_error_segment = -1;
    ruckig::Result first_error = ruckig::Result::Working;
    double max_duration = 0.0;
    double sum_duration = 0.0;

    for (std::size_t segment = 0; segment + 1 < points.size(); ++segment) {
      ruckig::InputParameter<ruckig::DynamicDOFs> input(kDofs);
      input.current_position = std::vector<double>(points[segment].q.begin(), points[segment].q.end());
      input.target_position = std::vector<double>(points[segment + 1].q.begin(), points[segment + 1].q.end());
      input.max_velocity = std::vector<double>(kMaxVelocity.begin(), kMaxVelocity.end());
      input.max_acceleration = std::vector<double>(kMaxAcceleration.begin(), kMaxAcceleration.end());
      input.max_jerk = std::vector<double>(kMaxJerk.begin(), kMaxJerk.end());
      input.min_position = std::vector<double>(kMinPosition.begin(), kMinPosition.end());
      input.max_position = std::vector<double>(kMaxPosition.begin(), kMaxPosition.end());
      input.current_velocity.resize(kDofs);
      input.target_velocity.resize(kDofs);
      input.current_acceleration.resize(kDofs);
      input.target_acceleration.resize(kDofs);
      for (std::size_t j = 0; j < kDofs; ++j) {
        input.current_velocity[j] = std::clamp(points[segment].v[j], -kMaxVelocity[j], kMaxVelocity[j]);
        input.target_velocity[j] = std::clamp(points[segment + 1].v[j], -kMaxVelocity[j], kMaxVelocity[j]);
        input.current_acceleration[j] = std::clamp(points[segment].a[j], -kMaxAcceleration[j], kMaxAcceleration[j]);
        input.target_acceleration[j] = std::clamp(points[segment + 1].a[j], -kMaxAcceleration[j], kMaxAcceleration[j]);
      }

      ruckig::Trajectory<ruckig::DynamicDOFs, ruckig::StandardVector> trajectory(kDofs);
      const auto result = ruckig.calculate(input, trajectory);
      const bool ok = result == ruckig::Result::Working || result == ruckig::Result::Finished;
      if (ok) {
        ++successful;
        max_duration = std::max(max_duration, trajectory.get_duration());
        sum_duration += trajectory.get_duration();
      } else if (result == ruckig::Result::ErrorPositionalLimits) {
        ++positional_errors;
      } else {
        ++other_errors;
      }
      if (!ok && first_error_segment < 0) { first_error_segment = static_cast<int>(segment); first_error = result; }

      output << "{\"segment\":" << segment << ",\"source_dt\":";
      number(output, points[segment + 1].t - points[segment].t);
      output << ",\"result\":{\"numeric\":" << static_cast<int>(result)
             << ",\"name\":\"" << result_name(result) << "\"},\"profile_available\":"
             << (ok ? "true" : "false") << ",\"duration\":";
      if (ok) number(output, trajectory.get_duration()); else output << "null";
      output << ",\"input_q0\":[";
      for (std::size_t j = 0; j < kDofs; ++j) { if (j) output << ','; number(output, points[segment].q[j]); }
      output << "],\"input_q1\":[";
      for (std::size_t j = 0; j < kDofs; ++j) { if (j) output << ','; number(output, points[segment + 1].q[j]); }
      output << "]}\n";
    }
    output.flush();
    std::cerr << "segments=" << points.size() - 1 << " successful=" << successful
              << " positional_errors=" << positional_errors << " other_errors=" << other_errors
              << " first_error_segment=" << first_error_segment;
    if (first_error_segment >= 0) std::cerr << " first_error=" << result_name(first_error);
    std::cerr << " max_duration=" << std::setprecision(17) << max_duration
              << " sum_duration=" << sum_duration << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 3;
  }
}
