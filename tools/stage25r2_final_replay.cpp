// Independent final RobotTrajectory replay for Stage 2.5R2.
//
// This executable is intentionally separate from MoveIt's smoothing loop.  It
// reads only the final exported q/dq/ddq/t, reconstructs one Ruckig input for
// every adjacent segment, and writes the native result/profile and the faithful
// 0.01 s overshoot check without modifying the trajectory.

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

namespace
{
using DynamicInput = ruckig::InputParameter<ruckig::DynamicDOFs>;
using DynamicTrajectory = ruckig::Trajectory<ruckig::DynamicDOFs, ruckig::StandardVector>;
constexpr size_t kDofs = 6;
constexpr double kOvershootCheckPeriod = 0.01;

struct Point
{
  double time { 0.0 };
  std::array<double, kDofs> q {};
  std::array<double, kDofs> dq {};
  std::array<double, kDofs> ddq {};
};

struct Overshoot
{
  bool positive { false };
  size_t joint_index { 0 };
  double local_time { 0.0 };
  double abs_error { 0.0 };
};

std::string json_string(const std::string& value)
{
  std::ostringstream out;
  out << '"';
  for (const char c : value)
  {
    switch (c)
    {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << c; break;
    }
  }
  out << '"';
  return out.str();
}

void json_double(std::ostream& out, const double value)
{
  if (!std::isfinite(value))
    out << "null";
  else
    out << std::setprecision(17) << value;
}

template <typename Vector>
void json_vector(std::ostream& out, const Vector& values)
{
  out << '[';
  for (size_t i = 0; i < values.size(); ++i)
  {
    if (i) out << ',';
    json_double(out, static_cast<double>(values.at(i)));
  }
  out << ']';
}

std::string result_name(const ruckig::Result result)
{
  switch (result)
  {
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

bool successful(const ruckig::Result result)
{
  return result == ruckig::Result::Working || result == ruckig::Result::Finished;
}

std::vector<double> split_doubles(const std::string& line)
{
  std::vector<double> values;
  std::stringstream stream(line);
  std::string token;
  while (std::getline(stream, token, ','))
  {
    try
    {
      values.push_back(std::stod(token));
    }
    catch (...)
    {
      return {};
    }
  }
  return values;
}

std::vector<Point> read_csv(const std::string& path)
{
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open final trajectory CSV: " + path);
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty final trajectory CSV");
  std::vector<Point> points;
  while (std::getline(input, line))
  {
    const auto values = split_doubles(line);
    if (values.empty()) continue;
    if (values.size() != 1 + 3 * kDofs + kDofs)
      throw std::runtime_error("unexpected final trajectory CSV column count");
    Point point;
    point.time = values[0];
    for (size_t i = 0; i < kDofs; ++i) point.q[i] = values[1 + i];
    for (size_t i = 0; i < kDofs; ++i) point.dq[i] = values[1 + kDofs + i];
    for (size_t i = 0; i < kDofs; ++i) point.ddq[i] = values[1 + 2 * kDofs + i];
    points.push_back(point);
  }
  return points;
}

void write_input(std::ostream& out, const DynamicInput& input)
{
  out << "{\"current_position\":"; json_vector(out, input.current_position);
  out << ",\"current_velocity\":"; json_vector(out, input.current_velocity);
  out << ",\"current_acceleration\":"; json_vector(out, input.current_acceleration);
  out << ",\"target_position\":"; json_vector(out, input.target_position);
  out << ",\"target_velocity\":"; json_vector(out, input.target_velocity);
  out << ",\"target_acceleration\":"; json_vector(out, input.target_acceleration);
  out << ",\"max_velocity\":"; json_vector(out, input.max_velocity);
  out << ",\"max_acceleration\":"; json_vector(out, input.max_acceleration);
  out << ",\"max_jerk\":"; json_vector(out, input.max_jerk);
  out << ",\"minimum_duration\":null,\"control_interface\":\"Position\",\"synchronization\":\"Time\"}";
}

void write_profile(std::ostream& out, const ruckig::Profile& profile)
{
  out << "{\"phase_duration\":"; json_vector(out, profile.t);
  out << ",\"cumulative_phase_boundaries\":"; json_vector(out, profile.t_sum);
  out << ",\"p\":"; json_vector(out, profile.p);
  out << ",\"v\":"; json_vector(out, profile.v);
  out << ",\"a\":"; json_vector(out, profile.a);
  out << ",\"j\":"; json_vector(out, profile.j);
  out << ",\"brake\":{\"duration\":"; json_double(out, profile.brake.duration);
  out << ",\"phase_duration\":"; json_vector(out, profile.brake.t);
  out << ",\"j\":"; json_vector(out, profile.brake.j);
  out << ",\"a\":"; json_vector(out, profile.brake.a);
  out << ",\"v\":"; json_vector(out, profile.brake.v);
  out << ",\"p\":"; json_vector(out, profile.brake.p);
  out << "}}";
}

double max_abs_jerk(const ruckig::Profile& profile)
{
  double value = 0.0;
  for (const double jerk : profile.j) value = std::max(value, std::abs(jerk));
  for (const double jerk : profile.brake.j) value = std::max(value, std::abs(jerk));
  return value;
}

Overshoot check_overshoot(DynamicTrajectory& output, const DynamicInput& input, const double threshold)
{
  Overshoot result;
  const auto profiles = output.get_profiles();
  for (double time = kOvershootCheckPeriod; time < output.get_duration(); time += kOvershootCheckPeriod)
  {
    std::vector<double> q(kDofs), dq(kDofs), ddq(kDofs);
    size_t section = 0;
    output.at_time(time, q, dq, ddq, section);
    (void)section;
    for (size_t joint = 0; joint < kDofs; ++joint)
    {
      const double current_minus_target = input.current_position[joint] - input.target_position[joint];
      const double sample_minus_target = q[joint] - input.target_position[joint];
      const double signed_ratio = sample_minus_target / current_minus_target;
      const double abs_error = std::fabs(sample_minus_target);
      if (signed_ratio < 0.0 && abs_error > threshold)
      {
        result.positive = true;
        if (result.abs_error == 0.0 || abs_error > result.abs_error)
        {
          result.joint_index = joint;
          result.local_time = time;
          result.abs_error = abs_error;
        }
      }
    }
  }
  return result;
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc != 4)
  {
    std::cerr << "usage: stage25r2_final_replay <trajectory.csv> <output.jsonl> <overshoot_threshold>\n";
    return 2;
  }
  try
  {
    const std::vector<Point> points = read_csv(argv[1]);
    if (points.size() < 2) throw std::runtime_error("final trajectory has fewer than two states");
    const double threshold = std::stod(argv[3]);
    std::ofstream output_file(argv[2], std::ios::out | std::ios::trunc);
    if (!output_file) throw std::runtime_error("cannot open replay output");

    ruckig::Ruckig<ruckig::DynamicDOFs> ruckig(kDofs, 0.01);
    for (size_t segment = 0; segment + 1 < points.size(); ++segment)
    {
      DynamicInput input(kDofs);
      input.current_position = std::vector<double>(points[segment].q.begin(), points[segment].q.end());
      input.target_position = std::vector<double>(points[segment + 1].q.begin(), points[segment + 1].q.end());
      input.max_velocity = { 0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48 };
      input.max_acceleration = { 0.105, 0.105, 0.105, 0.105, 0.105, 0.105 };
      input.max_jerk = { 8.0, 8.0, 8.0, 8.0, 8.0, 8.0 };
      // Match MoveIt 2.12.4 getNextRuckigInput(): positions are copied,
      // while velocity and acceleration endpoints are clamped to the native
      // limits before calculate().  The raw RobotTrajectory values remain
      // persisted in the exported CSV and in trajectory_state below.
      input.current_velocity.resize(kDofs);
      input.current_acceleration.resize(kDofs);
      input.target_velocity.resize(kDofs);
      input.target_acceleration.resize(kDofs);
      for (size_t joint = 0; joint < kDofs; ++joint)
      {
        input.current_velocity[joint] = std::clamp(points[segment].dq[joint], -input.max_velocity[joint], input.max_velocity[joint]);
        input.current_acceleration[joint] = std::clamp(points[segment].ddq[joint], -input.max_acceleration[joint], input.max_acceleration[joint]);
        input.target_velocity[joint] = std::clamp(points[segment + 1].dq[joint], -input.max_velocity[joint], input.max_velocity[joint]);
        input.target_acceleration[joint] = std::clamp(points[segment + 1].ddq[joint], -input.max_acceleration[joint], input.max_acceleration[joint]);
      }

      DynamicTrajectory trajectory(kDofs);
      const ruckig::Result result = ruckig.calculate(input, trajectory);
      const bool valid = successful(result);
      const auto profiles = trajectory.get_profiles();
      double maximum_jerk = 0.0;
      if (valid)
        for (size_t joint = 0; joint < kDofs; ++joint) maximum_jerk = std::max(maximum_jerk, max_abs_jerk(profiles[0].at(joint)));
      Overshoot overshoot;
      if (valid) overshoot = check_overshoot(trajectory, input, threshold);

      output_file << "{\"segment_index\":" << segment << ",\"exported_dt\":";
      json_double(output_file, points[segment + 1].time - points[segment].time);
      output_file << ",\"ruckig_result\":{\"numeric_result\":" << static_cast<int>(result)
                  << ",\"result_name\":" << json_string(result_name(result)) << "}"
                  << ",\"native_profiles_available\":" << (valid ? "true" : "false")
                  << ",\"native_duration\":";
      if (valid) json_double(output_file, trajectory.get_duration()); else output_file << "null";
      output_file << ",\"native_minimum_or_generated_duration\":";
      if (valid) json_double(output_file, trajectory.get_duration()); else output_file << "null";
      output_file << ",\"duration_identity\":";
      if (!valid) output_file << "\"unavailable\"";
      else output_file << (std::abs((points[segment + 1].time - points[segment].time) - trajectory.get_duration()) <= 1e-12 ? "\"equal\"" : "\"mismatch\"");
      output_file << ",\"input\":"; write_input(output_file, input);
      output_file << ",\"trajectory_state\":{\"q0\":"; json_vector(output_file, points[segment].q);
      output_file << ",\"dq0\":"; json_vector(output_file, points[segment].dq);
      output_file << ",\"ddq0\":"; json_vector(output_file, points[segment].ddq);
      output_file << ",\"q1\":"; json_vector(output_file, points[segment + 1].q);
      output_file << ",\"dq1\":"; json_vector(output_file, points[segment + 1].dq);
      output_file << ",\"ddq1\":"; json_vector(output_file, points[segment + 1].ddq);
      output_file << "}";
      output_file << ",\"profile\":";
      if (!valid) output_file << "null";
      else
      {
        output_file << "[";
        for (size_t joint = 0; joint < kDofs; ++joint)
        {
          if (joint) output_file << ',';
          write_profile(output_file, profiles[0].at(joint));
        }
        output_file << "]";
      }
      output_file << ",\"max_abs_jerk\":";
      if (valid) json_double(output_file, maximum_jerk); else output_file << "null";
      output_file << ",\"overshoot_at_selected_semantics\":{\"threshold\":";
      json_double(output_file, threshold);
      output_file << ",\"passed\":" << (valid && !overshoot.positive ? "true" : "false")
                  << ",\"positive\":" << (overshoot.positive ? "true" : "false")
                  << ",\"worst_joint_index\":" << overshoot.joint_index << ",\"local_time\":";
      json_double(output_file, overshoot.local_time);
      output_file << ",\"abs_overshoot_rad\":";
      json_double(output_file, overshoot.abs_error);
      output_file << "}}\n";
    }
    output_file.flush();
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 3;
  }
}
