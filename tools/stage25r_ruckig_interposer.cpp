// Stage 2.5R additive native instrumentation for MoveIt 2.12.4 / Ruckig 0.9.2.
//
// This shared object interposes only the private RuckigSmoothing::runRuckig
// member.  The implementation below follows MoveIt's 2.12.4 runRuckig loop
// and calls MoveIt's original private helpers for input preparation,
// overshoot checking, and duration extension.  It records the actual runtime
// InputParameter immediately after calculate() returns, before overshoot
// handling, and never changes the formal Stage 2.5 artifacts.

#include <moveit/robot_trajectory/robot_trajectory.hpp>
#define private public
#include <moveit/trajectory_processing/ruckig_traj_smoothing.hpp>
#undef private
#include <moveit/trajectory_processing/trajectory_tools.hpp>
#include <ruckig/ruckig.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "certified_output_guard.hpp"

namespace
{
using DynamicInput = ruckig::InputParameter<ruckig::DynamicDOFs>;
using DynamicTrajectory = ruckig::Trajectory<ruckig::DynamicDOFs, ruckig::StandardVector>;

constexpr double kMaxDurationExtensionFactor = 50.0;
constexpr double kDurationExtensionFraction = 1.1;
constexpr double kOvershootCheckPeriod = 0.01;

bool env_flag(const char* name)
{
  const char* value = std::getenv(name);
  return value && (*value == '1' || *value == 'y' || *value == 'Y' || *value == 't' || *value == 'T');
}

bool compact_probe()
{
  static const bool value = env_flag("H12_R5_COMPACT_PROBE");
  return value;
}

void json_double(std::ostream& out, double value);

template <typename Vector>
void json_double_vector(std::ostream& out, const Vector& values, size_t count);

struct H12R5AHelperSnapshot
{
  std::vector<double> durations;
  std::vector<std::vector<double>> positions;
  std::vector<std::vector<double>> velocities;
  std::vector<std::vector<double>> accelerations;
};

H12R5AHelperSnapshot capture_h12_r5a_helper_snapshot(
    const robot_trajectory::RobotTrajectory& trajectory, const std::vector<int>& move_group_idx)
{
  H12R5AHelperSnapshot snapshot;
  const size_t count = trajectory.getWayPointCount();
  snapshot.durations.reserve(count);
  snapshot.positions.reserve(count);
  snapshot.velocities.reserve(count);
  snapshot.accelerations.reserve(count);
  for (size_t waypoint = 0; waypoint < count; ++waypoint)
  {
    snapshot.durations.push_back(trajectory.getWayPointDurationFromPrevious(waypoint));
    const auto& state = trajectory.getWayPoint(waypoint);
    std::vector<double> positions, velocities, accelerations;
    positions.reserve(move_group_idx.size());
    velocities.reserve(move_group_idx.size());
    accelerations.reserve(move_group_idx.size());
    for (const int variable : move_group_idx)
    {
      positions.push_back(state.getVariablePosition(variable));
      velocities.push_back(state.getVariableVelocity(variable));
      accelerations.push_back(state.getVariableAcceleration(variable));
    }
    snapshot.positions.push_back(std::move(positions));
    snapshot.velocities.push_back(std::move(velocities));
    snapshot.accelerations.push_back(std::move(accelerations));
  }
  return snapshot;
}

std::vector<size_t> h12_r5a_changed_scalar_indices(const std::vector<double>& before,
                                                   const std::vector<double>& after)
{
  std::vector<size_t> changed;
  for (size_t index = 0; index < std::min(before.size(), after.size()); ++index)
  {
    if (std::abs(before.at(index) - after.at(index)) > 1.0e-14)
      changed.push_back(index);
  }
  return changed;
}

std::vector<size_t> h12_r5a_changed_matrix_indices(const std::vector<std::vector<double>>& before,
                                                   const std::vector<std::vector<double>>& after)
{
  std::vector<size_t> changed;
  for (size_t waypoint = 0; waypoint < std::min(before.size(), after.size()); ++waypoint)
  {
    bool differs = before.at(waypoint).size() != after.at(waypoint).size();
    for (size_t joint = 0; !differs && joint < before.at(waypoint).size(); ++joint)
      differs = std::abs(before.at(waypoint).at(joint) - after.at(waypoint).at(joint)) > 1.0e-14;
    if (differs)
      changed.push_back(waypoint);
  }
  return changed;
}

void h12_r5a_write_size_vector(std::ostream& out, const std::vector<size_t>& values)
{
  out << '[';
  for (size_t index = 0; index < values.size(); ++index)
  {
    if (index) out << ',';
    out << values.at(index);
  }
  out << ']';
}

void h12_r5a_write_matrix(std::ostream& out, const std::vector<std::vector<double>>& values)
{
  out << '[';
  for (size_t index = 0; index < values.size(); ++index)
  {
    if (index) out << ',';
    json_double_vector(out, values.at(index), values.at(index).size());
  }
  out << ']';
}

void h12_r5a_write_helper_snapshot(std::ostream& out, const H12R5AHelperSnapshot& snapshot)
{
  out << "{\"duration_from_previous\":";
  json_double_vector(out, snapshot.durations, snapshot.durations.size());
  out << ",\"position\":";
  h12_r5a_write_matrix(out, snapshot.positions);
  out << ",\"velocity\":";
  h12_r5a_write_matrix(out, snapshot.velocities);
  out << ",\"acceleration\":";
  h12_r5a_write_matrix(out, snapshot.accelerations);
  out << '}';
}

bool h12_r5a_helper_mutation_invariant(const H12R5AHelperSnapshot& before,
                                       const H12R5AHelperSnapshot& after,
                                       const H12R5AHelperSnapshot& original,
                                       const size_t waypoint_idx, const double factor)
{
  const size_t target = waypoint_idx + 1;
  if (target >= before.durations.size() || before.durations.size() != after.durations.size()) return false;
  const auto duration_changed = h12_r5a_changed_scalar_indices(before.durations, after.durations);
  if (duration_changed != std::vector<size_t>{ target }) return false;
  if (!h12_r5a_changed_matrix_indices(before.positions, after.positions).empty()) return false;
  const auto velocity_changed = h12_r5a_changed_matrix_indices(before.velocities, after.velocities);
  const auto acceleration_changed = h12_r5a_changed_matrix_indices(before.accelerations, after.accelerations);
  if (std::any_of(velocity_changed.begin(), velocity_changed.end(), [target](size_t index) { return index != target; }))
    return false;
  if (std::any_of(acceleration_changed.begin(), acceleration_changed.end(), [target](size_t index) { return index != target; }))
    return false;
  const double expected_duration = factor * original.durations.at(target);
  if (std::abs(after.durations.at(target) - expected_duration) > 1.0e-10) return false;
  for (size_t joint = 0; joint < after.velocities.at(target).size(); ++joint)
  {
    const double expected_velocity = before.velocities.at(target).at(joint) / factor;
    const double actual_velocity = after.velocities.at(target).at(joint);
    if (std::abs(actual_velocity - expected_velocity) > 1.0e-10) return false;
    const double expected_acceleration =
        (actual_velocity - after.velocities.at(waypoint_idx).at(joint)) / after.durations.at(target);
    if (std::abs(after.accelerations.at(target).at(joint) - expected_acceleration) > 1.0e-10) return false;
  }
  return true;
}

void run_h12_r5a_installed_helper_probe(const robot_trajectory::RobotTrajectory& source,
                                        const moveit::core::JointModelGroup* group)
{
  if (!env_flag("H12_R5A_HELPER_PROBE")) return;
  static bool written = false;
  if (written) return;
  written = true;

  constexpr size_t waypoint_count = 6;
  constexpr double factor = 1.1;
  const size_t num_dof = group->getVariableCount();
  const std::vector<int>& move_group_idx = group->getVariableIndexList();
  robot_trajectory::RobotTrajectory original(source.getRobotModel(), group);
  const auto& seed = source.getFirstWayPoint();
  for (size_t waypoint = 0; waypoint < waypoint_count; ++waypoint)
  {
    moveit::core::RobotState state(seed);
    for (size_t joint = 0; joint < num_dof; ++joint)
    {
      const int variable = move_group_idx.at(joint);
      state.setVariableVelocity(variable, 0.01 * static_cast<double>((waypoint + 1) * (joint + 1)));
      state.setVariableAcceleration(variable, -0.02 * static_cast<double>((waypoint + 1) * (joint + 1)));
    }
    original.addSuffixWayPoint(state, waypoint == 0 ? 0.0 : 0.10 + 0.03 * static_cast<double>(waypoint));
  }

  const char* raw_dir = std::getenv("STAGE25R_NATIVE_DIR");
  if (!raw_dir || !*raw_dir) return;
  certified_output_guard::assert_output_path_writable(std::filesystem::path(raw_dir));
  std::ofstream stream(std::filesystem::path(raw_dir) / "h12_r5a_extend_duration_binary_probe_raw.jsonl",
                       std::ios::out | std::ios::binary | std::ios::trunc);
  for (const size_t second_argument : { size_t{ 2 }, size_t{ 1 } })
  {
    robot_trajectory::RobotTrajectory mutated(original, true);
    const H12R5AHelperSnapshot before = capture_h12_r5a_helper_snapshot(mutated, move_group_idx);
    trajectory_processing::RuckigSmoothing::extendTrajectoryDuration(
        factor, second_argument, num_dof, move_group_idx, original, mutated);
    const H12R5AHelperSnapshot after = capture_h12_r5a_helper_snapshot(mutated, move_group_idx);
    const auto duration_changed = h12_r5a_changed_scalar_indices(before.durations, after.durations);
    const auto position_changed = h12_r5a_changed_matrix_indices(before.positions, after.positions);
    const auto velocity_changed = h12_r5a_changed_matrix_indices(before.velocities, after.velocities);
    const auto acceleration_changed = h12_r5a_changed_matrix_indices(before.accelerations, after.accelerations);
    stream << "{\"schema_version\":\"stage3_h12_r5a_installed_helper_probe_raw_v1\",\"factor\":";
    json_double(stream, factor);
    stream << ",\"second_argument\":" << second_argument << ",\"num_waypoints\":" << waypoint_count
           << ",\"num_dof\":" << num_dof << ",\"before\":";
    h12_r5a_write_helper_snapshot(stream, before);
    stream << ",\"after\":";
    h12_r5a_write_helper_snapshot(stream, after);
    stream << ",\"duration_changed_indices\":";
    h12_r5a_write_size_vector(stream, duration_changed);
    stream << ",\"position_changed_indices\":";
    h12_r5a_write_size_vector(stream, position_changed);
    stream << ",\"velocity_changed_indices\":";
    h12_r5a_write_size_vector(stream, velocity_changed);
    stream << ",\"acceleration_changed_indices\":";
    h12_r5a_write_size_vector(stream, acceleration_changed);
    stream << "}\n";
  }
  stream.flush();
}

__attribute__((constructor)) void stage25r_library_loaded()
{
  const char* raw_dir = std::getenv("STAGE25R_NATIVE_DIR");
  if (!raw_dir || !*raw_dir) return;
  certified_output_guard::assert_output_path_writable(std::filesystem::path(raw_dir));
  std::error_code error;
  std::filesystem::create_directories(std::filesystem::path(raw_dir), error);
  std::ofstream marker(std::filesystem::path(raw_dir) / "stage25r_library_loaded.marker", std::ios::out | std::ios::trunc);
  marker << "loaded\n";
}

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
  {
    out << "null";
    return;
  }
  out << std::setprecision(17) << value;
}

template <typename Vector>
void json_double_vector(std::ostream& out, const Vector& values, const size_t count)
{
  out << '[';
  for (size_t i = 0; i < count; ++i)
  {
    if (i) out << ',';
    json_double(out, static_cast<double>(values.at(i)));
  }
  out << ']';
}

template <typename Vector>
void json_bool_vector(std::ostream& out, const Vector& values, const size_t count)
{
  out << '[';
  for (size_t i = 0; i < count; ++i)
  {
    if (i) out << ',';
    out << (static_cast<bool>(values.at(i)) ? "true" : "false");
  }
  out << ']';
}

template <typename Vector>
void json_string_vector(std::ostream& out, const Vector& values, const size_t count,
                        const std::string (*name)(const typename Vector::value_type&))
{
  out << '[';
  for (size_t i = 0; i < count; ++i)
  {
    if (i) out << ',';
    out << json_string(name(values.at(i)));
  }
  out << ']';
}

std::string control_interface_name(const ruckig::ControlInterface value)
{
  return value == ruckig::ControlInterface::Velocity ? "Velocity" : "Position";
}

std::string synchronization_name(const ruckig::Synchronization value)
{
  switch (value)
  {
    case ruckig::Synchronization::Time: return "Time";
    case ruckig::Synchronization::TimeIfNecessary: return "TimeIfNecessary";
    case ruckig::Synchronization::Phase: return "Phase";
    case ruckig::Synchronization::None: return "None";
  }
  return "unknown";
}

std::string duration_discretization_name(const ruckig::DurationDiscretization value)
{
  return value == ruckig::DurationDiscretization::Discrete ? "Discrete" : "Continuous";
}

std::string result_name(const ruckig::Result value)
{
  switch (value)
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

bool successful(const ruckig::Result value)
{
  return value == ruckig::Result::Working || value == ruckig::Result::Finished;
}

template <typename Vector>
void json_optional_double_vector(std::ostream& out, const std::optional<Vector>& values, const size_t count)
{
  if (!values)
  {
    out << "null";
    return;
  }
  json_double_vector(out, values.value(), count);
}

template <typename Enum>
void json_optional_enum_vector(std::ostream& out, const std::optional<std::vector<Enum>>& values,
                              const size_t count, const std::string (*name)(const Enum&))
{
  if (!values)
  {
    out << "null";
    return;
  }
  out << '[';
  for (size_t i = 0; i < count; ++i)
  {
    if (i) out << ',';
    out << json_string(name(values.value().at(i)));
  }
  out << ']';
}

void write_brake(std::ostream& out, const ruckig::BrakeProfile& brake)
{
  out << "{\"duration\":";
  json_double(out, brake.duration);
  out << ",\"phase_duration\":[";
  for (size_t i = 0; i < brake.t.size(); ++i)
  {
    if (i) out << ',';
    json_double(out, brake.t[i]);
  }
  out << "],\"j\":[";
  for (size_t i = 0; i < brake.j.size(); ++i)
  {
    if (i) out << ',';
    json_double(out, brake.j[i]);
  }
  out << "],\"a\":[";
  for (size_t i = 0; i < brake.a.size(); ++i)
  {
    if (i) out << ',';
    json_double(out, brake.a[i]);
  }
  out << "],\"v\":[";
  for (size_t i = 0; i < brake.v.size(); ++i)
  {
    if (i) out << ',';
    json_double(out, brake.v[i]);
  }
  out << "],\"p\":[";
  for (size_t i = 0; i < brake.p.size(); ++i)
  {
    if (i) out << ',';
    json_double(out, brake.p[i]);
  }
  out << "]}";
}

void write_input(std::ostream& out, const DynamicInput& input, const size_t dofs)
{
  out << "{\"degrees_of_freedom\":" << dofs;
  out << ",\"current_position\":"; json_double_vector(out, input.current_position, dofs);
  out << ",\"current_velocity\":"; json_double_vector(out, input.current_velocity, dofs);
  out << ",\"current_acceleration\":"; json_double_vector(out, input.current_acceleration, dofs);
  out << ",\"target_position\":"; json_double_vector(out, input.target_position, dofs);
  out << ",\"target_velocity\":"; json_double_vector(out, input.target_velocity, dofs);
  out << ",\"target_acceleration\":"; json_double_vector(out, input.target_acceleration, dofs);
  out << ",\"max_velocity\":"; json_double_vector(out, input.max_velocity, dofs);
  out << ",\"max_acceleration\":"; json_double_vector(out, input.max_acceleration, dofs);
  out << ",\"max_jerk\":"; json_double_vector(out, input.max_jerk, dofs);
  out << ",\"min_velocity\":"; json_optional_double_vector(out, input.min_velocity, dofs);
  out << ",\"min_acceleration\":"; json_optional_double_vector(out, input.min_acceleration, dofs);
  out << ",\"enabled\":"; json_bool_vector(out, input.enabled, dofs);
  out << ",\"control_interface\":" << json_string(control_interface_name(input.control_interface));
  out << ",\"synchronization\":" << json_string(synchronization_name(input.synchronization));
  out << ",\"duration_discretization\":" << json_string(duration_discretization_name(input.duration_discretization));
  out << ",\"per_dof_control_interface\":";
  if (!input.per_dof_control_interface)
    out << "null";
  else
  {
    out << '[';
    for (size_t i = 0; i < dofs; ++i)
    {
      if (i) out << ',';
      out << json_string(control_interface_name(input.per_dof_control_interface.value().at(i)));
    }
    out << ']';
  }
  out << ",\"per_dof_synchronization\":";
  if (!input.per_dof_synchronization)
    out << "null";
  else
  {
    out << '[';
    for (size_t i = 0; i < dofs; ++i)
    {
      if (i) out << ',';
      out << json_string(synchronization_name(input.per_dof_synchronization.value().at(i)));
    }
    out << ']';
  }
  out << ",\"minimum_duration\":";
  if (input.minimum_duration) json_double(out, input.minimum_duration.value()); else out << "null";
  out << ",\"intermediate_positions\":[";
  for (size_t i = 0; i < input.intermediate_positions.size(); ++i)
  {
    if (i) out << ',';
    json_double_vector(out, input.intermediate_positions.at(i), dofs);
  }
  out << "]";
  out << ",\"per_section_max_velocity\":null,\"per_section_max_acceleration\":null,\"per_section_max_jerk\":null";
  out << ",\"per_section_min_velocity\":null,\"per_section_min_acceleration\":null";
  out << ",\"max_position\":null,\"min_position\":null,\"per_section_minimum_duration\":null";
  out << "}";
}

struct KinematicExtrema
{
  double max_abs_velocity { 0.0 };
  double max_abs_acceleration { 0.0 };
  double max_abs_jerk { 0.0 };
};

struct ExactPositionExtrema
{
  double min { std::numeric_limits<double>::infinity() };
  double max { -std::numeric_limits<double>::infinity() };
  double t_min { 0.0 };
  double t_max { 0.0 };
};

struct OvershootPoint
{
  size_t joint_index { 0 };
  double local_time { 0.0 };
  double current_position { 0.0 };
  double target_position { 0.0 };
  double sampled_position { 0.0 };
  double current_minus_target { 0.0 };
  double sample_minus_target { 0.0 };
  double signed_ratio { 0.0 };
  double abs_overshoot_rad { 0.0 };
  double overshoot_threshold_rad { 0.0 };
  double native_output_duration { 0.0 };
  int phase { -1 };
  size_t section { 0 };
  std::vector<double> q;
  std::vector<double> dq;
  std::vector<double> ddq;
  std::vector<double> native_jerk;
};

struct OvershootCheckResult
{
  bool overshoots { false };
  std::optional<OvershootPoint> first;
  std::optional<OvershootPoint> worst;
};

void update_extrema(KinematicExtrema& ext, const double p, const double v, const double a, const double j)
{
  (void)p;
  ext.max_abs_velocity = std::max(ext.max_abs_velocity, std::abs(v));
  ext.max_abs_acceleration = std::max(ext.max_abs_acceleration, std::abs(a));
  ext.max_abs_jerk = std::max(ext.max_abs_jerk, std::abs(j));
}

void scan_piece(KinematicExtrema& ext, const double duration, const double p0, const double v0,
                const double a0, const double j, const double /*unused*/)
{
  if (!(duration > 0.0) || !std::isfinite(duration)) return;
  auto eval = [&](const double t) {
    const double p = p0 + t * (v0 + t * (a0 / 2.0 + t * j / 6.0));
    const double v = v0 + t * (a0 + t * j / 2.0);
    const double a = a0 + t * j;
    update_extrema(ext, p, v, a, j);
  };
  eval(0.0);
  eval(duration);
  if (std::abs(j) > 1e-14)
  {
    const double ta = -a0 / j;
    if (ta > 0.0 && ta < duration) eval(ta);
    const double discriminant = a0 * a0 - 2.0 * j * v0;
    if (discriminant >= 0.0)
    {
      const double root = std::sqrt(discriminant);
      const double tv0 = (-a0 - root) / j;
      const double tv1 = (-a0 + root) / j;
      if (tv0 > 0.0 && tv0 < duration) eval(tv0);
      if (tv1 > 0.0 && tv1 < duration) eval(tv1);
    }
  }
}

KinematicExtrema kinematic_extrema(const ruckig::Profile& profile)
{
  KinematicExtrema ext;
  for (size_t i = 0; i < profile.brake.t.size(); ++i)
  {
    scan_piece(ext, profile.brake.t[i], profile.brake.p[i], profile.brake.v[i], profile.brake.a[i], profile.brake.j[i], 0.0);
  }
  for (size_t i = 0; i < profile.t.size(); ++i)
  {
    scan_piece(ext, profile.t[i], profile.p[i], profile.v[i], profile.a[i], profile.j[i], 0.0);
  }
  update_extrema(ext, profile.p.back(), profile.v.back(), profile.a.back(), 0.0);
  return ext;
}

void scan_position_piece(ExactPositionExtrema& ext, const double offset, const double duration,
                         const double p0, const double v0, const double a0, const double j)
{
  // Zero-duration phases contain unused/default profile slots in Ruckig 0.9.2
  // and must not contribute an endpoint candidate.
  if (!(duration > 0.0) || !std::isfinite(duration)) return;
  auto eval = [&](const double t) {
    const double p = p0 + t * (v0 + t * (a0 / 2.0 + t * j / 6.0));
    if (p < ext.min) { ext.min = p; ext.t_min = offset + t; }
    if (p > ext.max) { ext.max = p; ext.t_max = offset + t; }
  };
  eval(0.0);
  eval(duration);
  if (std::abs(j) > 1.0e-14)
  {
    const double discriminant = a0 * a0 - 2.0 * j * v0;
    if (discriminant >= 0.0)
    {
      const double root = std::sqrt(discriminant);
      for (const double candidate : { (-a0 - root) / j, (-a0 + root) / j })
        if (candidate > 0.0 && candidate < duration) eval(candidate);
    }
  }
  else if (std::abs(a0) > 1.0e-14)
  {
    // Ruckig 0.9.2 Profile::get_position_extrema() omits this constant-
    // acceleration velocity root because its implementation only searches
    // roots when jerk != 0.  The omission is material for the R4 events.
    const double candidate = -v0 / a0;
    if (candidate > 0.0 && candidate < duration) eval(candidate);
  }
}

ExactPositionExtrema exact_position_extrema(const ruckig::Profile& profile)
{
  ExactPositionExtrema ext;
  double offset = 0.0;
  for (size_t i = 0; i < profile.brake.t.size(); ++i)
  {
    scan_position_piece(ext, offset, profile.brake.t[i], profile.brake.p[i], profile.brake.v[i],
                        profile.brake.a[i], profile.brake.j[i]);
    offset += profile.brake.t[i];
  }
  offset = profile.brake.duration;
  for (size_t i = 0; i < profile.t.size(); ++i)
  {
    scan_position_piece(ext, offset, profile.t[i], profile.p[i], profile.v[i], profile.a[i], profile.j[i]);
    offset += profile.t[i];
  }
  if (profile.pf < ext.min) { ext.min = profile.pf; ext.t_min = offset; }
  if (profile.pf > ext.max) { ext.max = profile.pf; ext.t_max = offset; }
  return ext;
}

int phase_at(const ruckig::Profile& profile, const double time)
{
  if (!(time < profile.brake.duration))
  {
    const double main_time = time - profile.brake.duration;
    for (size_t i = 0; i < profile.t_sum.size(); ++i)
    {
      if (main_time < profile.t_sum[i]) return static_cast<int>(i);
    }
    return -1;
  }
  for (size_t i = 0; i < profile.brake.t.size(); ++i)
  {
    const double end = (i == 0) ? profile.brake.t[0] : profile.brake.t[0] + profile.brake.t[1];
    if (time < end) return -2 - static_cast<int>(i);
  }
  return -1;
}

double jerk_at(const ruckig::Profile& profile, const int phase)
{
  if (phase >= 0 && static_cast<size_t>(phase) < profile.j.size()) return profile.j.at(static_cast<size_t>(phase));
  if (phase <= -2)
  {
    const size_t index = static_cast<size_t>(-2 - phase);
    if (index < profile.brake.j.size()) return profile.brake.j.at(index);
  }
  return 0.0;
}

void write_overshoot_point(std::ostream& out, const OvershootPoint& point)
{
  out << "{\"joint_index\":" << point.joint_index << ",\"local_time\":";
  json_double(out, point.local_time);
  out << ",\"current_position\":";
  json_double(out, point.current_position);
  out << ",\"target_position\":";
  json_double(out, point.target_position);
  out << ",\"sampled_position\":";
  json_double(out, point.sampled_position);
  out << ",\"current_minus_target\":";
  json_double(out, point.current_minus_target);
  out << ",\"sample_minus_target\":";
  json_double(out, point.sample_minus_target);
  out << ",\"signed_ratio\":";
  json_double(out, point.signed_ratio);
  out << ",\"abs_overshoot_rad\":";
  json_double(out, point.abs_overshoot_rad);
  out << ",\"overshoot_threshold_rad\":";
  json_double(out, point.overshoot_threshold_rad);
  out << ",\"native_output_duration\":";
  json_double(out, point.native_output_duration);
  out << ",\"phase\":" << point.phase << ",\"section\":" << point.section;
  out << ",\"q\":";
  json_double_vector(out, point.q, point.q.size());
  out << ",\"dq\":";
  json_double_vector(out, point.dq, point.dq.size());
  out << ",\"ddq\":";
  json_double_vector(out, point.ddq, point.ddq.size());
  out << ",\"native_jerk\":";
  json_double_vector(out, point.native_jerk, point.native_jerk.size());
  out << "}";
}

void write_optional_overshoot_point(std::ostream& out, const std::optional<OvershootPoint>& point)
{
  if (point)
    write_overshoot_point(out, point.value());
  else
    out << "null";
}

class NativeRecorder
{
public:
  NativeRecorder()
  {
    const char* raw_dir = std::getenv("STAGE25R_NATIVE_DIR");
    if (!raw_dir || !*raw_dir) return;
    directory_ = std::filesystem::path(raw_dir);
    certified_output_guard::assert_output_path_writable(directory_);
    enabled_ = true;
    std::filesystem::create_directories(directory_);
    calls_.open(directory_ / "stage25r_ruckig_calls.jsonl", std::ios::out | std::ios::binary | std::ios::trunc);
    resolutions_.open(directory_ / "stage25r_ruckig_call_resolutions.jsonl", std::ios::out | std::ios::binary | std::ios::trunc);
    samples_.open(directory_ / "stage25r_native_samples.jsonl", std::ios::out | std::ios::binary | std::ios::trunc);
    overshoot_events_.open(directory_ / "stage25r2_overshoot_events.jsonl", std::ios::out | std::ios::binary | std::ios::trunc);
    generation_events_.open(directory_ / "stage25r2_trajectory_generation_events.jsonl", std::ios::out | std::ios::binary | std::ios::trunc);
    remediation_events_.open(directory_ / "stage3_h12_r5_remediation_events.jsonl", std::ios::out | std::ios::binary | std::ios::trunc);
    extension_calls_.open(directory_ / "h12_r5a_extension_calls_raw.jsonl", std::ios::out | std::ios::binary | std::ios::trunc);
    std::ofstream marker(directory_ / "stage25r_instrumentation_marker.json", std::ios::out | std::ios::trunc);
    marker << "{\"status\":\"loaded\",\"instrumentation\":\"MoveIt_2.12.4_RuckigSmoothing_runRuckig_symbol_interposition\",\"ruckig_target\":\"0.9.2\",\"calculate_capture_point\":\"immediately_after_return_before_overshoot_check\"}\n";
    generation_events_ << "{\"schema_version\":\"stage25r2-trajectory-generation-v1\",\"event\":\"initial_generation\",\"trajectory_generation\":0}\n";
    generation_events_.flush();
  }

  bool enabled() const { return enabled_; }

  void write_extension_call(const size_t global_call_index, const size_t waypoint_idx,
                            const double duration_extension_factor, const size_t second_argument_passed,
                            const H12R5AHelperSnapshot& original, const H12R5AHelperSnapshot& before,
                            const H12R5AHelperSnapshot& after, const bool invariant_passed)
  {
    if (!enabled_) return;
    const auto duration_changed = h12_r5a_changed_scalar_indices(before.durations, after.durations);
    const auto position_changed = h12_r5a_changed_matrix_indices(before.positions, after.positions);
    const auto velocity_changed = h12_r5a_changed_matrix_indices(before.velocities, after.velocities);
    const auto acceleration_changed = h12_r5a_changed_matrix_indices(before.accelerations, after.accelerations);
    extension_calls_ << "{\"schema_version\":\"stage3_h12_r5a_extension_call_raw_v1\",\"global_call_index\":"
                     << global_call_index << ",\"trajectory_invocation_index\":" << run_summary_index_
                     << ",\"waypoint_idx\":" << waypoint_idx << ",\"duration_extension_factor\":";
    json_double(extension_calls_, duration_extension_factor);
    extension_calls_ << ",\"second_argument_passed\":" << second_argument_passed
                     << ",\"num_waypoints\":" << before.durations.size()
                     << ",\"original_duration_from_previous\":";
    json_double(extension_calls_, original.durations.at(waypoint_idx + 1));
    extension_calls_ << ",\"before\":";
    h12_r5a_write_helper_snapshot(extension_calls_, before);
    extension_calls_ << ",\"after\":";
    h12_r5a_write_helper_snapshot(extension_calls_, after);
    extension_calls_ << ",\"changed_duration_indices\":";
    h12_r5a_write_size_vector(extension_calls_, duration_changed);
    extension_calls_ << ",\"changed_position_indices\":";
    h12_r5a_write_size_vector(extension_calls_, position_changed);
    extension_calls_ << ",\"changed_velocity_indices\":";
    h12_r5a_write_size_vector(extension_calls_, velocity_changed);
    extension_calls_ << ",\"changed_acceleration_indices\":";
    h12_r5a_write_size_vector(extension_calls_, acceleration_changed);
    extension_calls_ << ",\"helper_mutation_invariant\":"
                     << json_string(invariant_passed ? "PASSED" : "FAILED") << "}\n";
    extension_calls_.flush();
  }

  void write_remediation(const size_t call_index, const size_t waypoint_idx, const size_t joint_index,
                         const double before_velocity, const double after_velocity, const double factor,
                         const char* boundary_state)
  {
    if (!enabled_) return;
    remediation_events_ << "{\"schema_version\":\"stage3-h12-r5-boundary-velocity-v1\",\"global_calculate_call_index\":"
                        << call_index << ",\"waypoint_idx\":" << waypoint_idx << ",\"joint_index\":" << joint_index
                        << ",\"boundary_state\":" << json_string(boundary_state)
                        << ",\"before_velocity\":";
    json_double(remediation_events_, before_velocity);
    remediation_events_ << ",\"after_velocity\":";
    json_double(remediation_events_, after_velocity);
    remediation_events_ << ",\"scale_factor\":";
    json_double(remediation_events_, factor);
    remediation_events_ << ",\"position_changed\":false,\"target_validation_after\":true}\n";
    if (!compact_probe()) remediation_events_.flush();
  }

  void write_immediate(const size_t call_index, const size_t waypoint_idx, const size_t attempt_index,
                       const size_t trajectory_generation, const double duration_extension_factor,
                       const double segment_duration_before,
                       const DynamicInput& input_before, const DynamicInput& input_after,
                       const DynamicTrajectory& output, const ruckig::Result result,
                       const moveit::core::JointModelGroup* group)
  {
    if (!enabled_) return;
    const size_t dofs = input_before.degrees_of_freedom;
    // Ruckig 0.9.2 validation is independent of the control-cycle value.  Run
    // all three H7.4-required validation modes on the exact input captured at
    // the calculate boundary, before any MoveIt retry/overshoot mutation.
    ruckig::Ruckig<ruckig::DynamicDOFs> validator(dofs, 0.01);
    const bool current_state_valid = validator.validate_input(input_before, true, false);
    const bool target_state_valid = validator.validate_input(input_before, false, true);
    const bool strict_state_valid = validator.validate_input(input_before, true, true);
    calls_ << "{\"schema_version\":\"stage25r-native-ruckig-call-v1\",\"global_calculate_call_index\":" << call_index
           << ",\"waypoint_idx\":" << waypoint_idx << ",\"segment_index\":" << waypoint_idx
           << ",\"attempt_index\":" << attempt_index << ",\"trajectory_generation\":" << trajectory_generation
           << ",\"duration_extension_factor\":";
    json_double(calls_, duration_extension_factor);
    calls_ << ",\"captured_immediately_before_calculate\":true,\"captured_immediately_after_calculate\":true,\"robot_trajectory_segment_duration_before\":";
    json_double(calls_, segment_duration_before);
    calls_ << ",\"input\":";
    write_input(calls_, input_before, dofs);
    calls_ << ",\"native_validate_input\":{\"current_state\":"
           << (current_state_valid ? "true" : "false")
           << ",\"target_state\":" << (target_state_valid ? "true" : "false")
           << ",\"current_and_target_strict\":" << (strict_state_valid ? "true" : "false")
           << "},\"input_after_calculate\":";
    if (compact_probe()) calls_ << "null"; else write_input(calls_, input_after, dofs);
    calls_ << ",\"result\":{\"numeric_result\":" << static_cast<int>(result) << ",\"result_name\":"
           << json_string(result_name(result)) << "},\"robot_model_bounds\":";
    write_model_bounds(calls_, group);
    calls_ << ",\"native_output\":";
    write_output(calls_, output, successful(result), dofs);
    calls_ << "}\n";
    if (!compact_probe()) calls_.flush();
  }

  void write_resolution(const size_t call_index, const size_t waypoint_idx, const size_t attempt_index,
                        const size_t trajectory_generation, const double duration_extension_factor,
                        const OvershootCheckResult& overshoot_result, const bool accepted,
                        const bool retried, const double segment_duration_after, const std::string& duration_source,
                        const bool final_segment_assignment, const double native_duration)
  {
    if (!enabled_) return;
    resolutions_ << "{\"schema_version\":\"stage25r-native-ruckig-resolution-v1\",\"global_calculate_call_index\":"
                 << call_index << ",\"waypoint_idx\":" << waypoint_idx << ",\"segment_index\":" << waypoint_idx
                 << ",\"attempt_index\":" << attempt_index << ",\"trajectory_generation\":" << trajectory_generation
                 << ",\"duration_extension_factor\":";
    json_double(resolutions_, duration_extension_factor);
    resolutions_ << ",\"overshoot\":" << (overshoot_result.overshoots ? "true" : "false")
                 << ",\"accepted_segment_call\":" << (accepted ? "true" : "false")
                 << ",\"accepted_in_generation\":";
    if (accepted) resolutions_ << trajectory_generation; else resolutions_ << "null";
    resolutions_ << ",\"retry_call\":" << (retried ? "true" : "false")
                 << ",\"first_overshoot\":";
    write_optional_overshoot_point(resolutions_, overshoot_result.first);
    resolutions_ << ",\"worst_overshoot_in_call\":";
    write_optional_overshoot_point(resolutions_, overshoot_result.worst);
    resolutions_ << ",\"robot_trajectory_segment_duration_after\":";
    json_double(resolutions_, segment_duration_after);
    resolutions_ << ",\"duration_source\":" << json_string(duration_source)
                 << ",\"changed_by_final_segment_assignment\":" << (final_segment_assignment ? "true" : "false")
                 << ",\"native_ruckig_output_duration\":";
    json_double(resolutions_, native_duration);
    resolutions_ << "}\n";
    if (!compact_probe()) resolutions_.flush();
    if (accepted)
      accepted_call_indices_.push_back(call_index);
  }

  void write_overshoot_event(const size_t call_index, const size_t waypoint_idx, const size_t attempt_index,
                             const size_t trajectory_generation, const double duration_extension_factor,
                             const OvershootCheckResult& overshoot_result)
  {
    if (!enabled_ || !overshoot_result.overshoots) return;
    overshoot_events_ << "{\"schema_version\":\"stage25r2-overshoot-event-v1\",\"global_calculate_call_index\":"
                      << call_index << ",\"waypoint_idx\":" << waypoint_idx << ",\"attempt_index\":" << attempt_index
                      << ",\"trajectory_generation\":" << trajectory_generation << ",\"duration_extension_factor\":";
    json_double(overshoot_events_, duration_extension_factor);
    overshoot_events_ << ",\"overshoot_check_period\":";
    json_double(overshoot_events_, kOvershootCheckPeriod);
    overshoot_events_ << ",\"first_overshoot\":";
    write_optional_overshoot_point(overshoot_events_, overshoot_result.first);
    overshoot_events_ << ",\"worst_overshoot_in_call\":";
    write_optional_overshoot_point(overshoot_events_, overshoot_result.worst);
    overshoot_events_ << "}\n";
    if (!compact_probe()) overshoot_events_.flush();
  }

  void write_generation_reset(const size_t from_generation, const size_t to_generation,
                              const size_t waypoint_idx, const size_t call_index)
  {
    if (!enabled_) return;
    std::vector<size_t> newly_invalidated;
    for (const size_t accepted_call : accepted_call_indices_)
    {
      if (stale_accepted_call_indices_.insert(accepted_call).second)
        newly_invalidated.push_back(accepted_call);
    }
    generation_events_ << "{\"schema_version\":\"stage25r2-trajectory-generation-v1\",\"event\":\"trajectory_reset\",\"from_generation\":"
                       << from_generation << ",\"to_generation\":" << to_generation << ",\"waypoint_idx\":" << waypoint_idx
                       << ",\"trigger_calculate_call_index\":" << call_index << ",\"invalidated_accepted_call_indices\":[";
    for (size_t i = 0; i < newly_invalidated.size(); ++i)
    {
      if (i) generation_events_ << ',';
      generation_events_ << newly_invalidated[i];
    }
    generation_events_ << "]}\n";
    generation_events_.flush();
    ++generation_reset_count_;
  }

  void write_samples(const size_t call_index, const size_t segment_index, const size_t attempt_index,
                     const DynamicTrajectory& output)
  {
    if (!enabled_ || compact_probe()) return;
    const size_t dofs = output.degrees_of_freedom;
    const double duration = output.get_duration();
    auto profiles = output.get_profiles();
    std::vector<std::pair<double, std::string>> times;
    auto add_time = [&](const double time, const std::string& side) {
      if (!std::isfinite(time)) return;
      const double clamped = std::max(0.0, std::min(duration, time));
      times.emplace_back(clamped, side);
    };
    add_time(0.0, "endpoint");
    add_time(duration, "endpoint");
    for (double t = 0.0; t < duration; t += kOvershootCheckPeriod) add_time(t, "overshoot_grid");
    for (size_t dof = 0; dof < dofs; ++dof)
    {
      const auto& p = profiles[0].at(dof);
      double base = 0.0;
      for (size_t i = 0; i < p.brake.t.size(); ++i)
      {
        if (p.brake.t[i] > 0.0) add_time(base + 0.5 * p.brake.t[i], "phase_interior");
        base += p.brake.t[i];
        if (i + 1 < p.brake.t.size() && p.brake.t[i] > 0.0 && p.brake.t[i + 1] > 0.0)
        {
          add_time(base, "phase_boundary_exact_right");
          add_time(std::nextafter(base, 0.0), "phase_boundary_left");
          add_time(std::nextafter(base, duration), "phase_boundary_right");
        }
      }
      for (size_t i = 0; i < p.t.size(); ++i)
      {
        const double phase_start = p.brake.duration + (i == 0 ? 0.0 : p.t_sum[i - 1]);
        const double phase_end = p.brake.duration + p.t_sum[i];
        if (p.t[i] > 0.0) add_time(0.5 * (phase_start + phase_end), "phase_interior");
        if (i + 1 < p.t.size() && p.t[i] > 0.0 && p.t[i + 1] > 0.0)
        {
          add_time(phase_end, "phase_boundary_exact_right");
          add_time(std::nextafter(phase_end, 0.0), "phase_boundary_left");
          add_time(std::nextafter(phase_end, duration), "phase_boundary_right");
        }
      }
    }
    std::sort(times.begin(), times.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.first != rhs.first) return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    });
    std::vector<double> q(dofs), dq(dofs), ddq(dofs);
    for (const auto& sample : times)
    {
      size_t section = 0;
      output.at_time(sample.first, q, dq, ddq, section);
       samples_ << "{\"schema_version\":\"stage25r-native-sample-v1\",\"global_calculate_call_index\":" << call_index
                << ",\"segment_index\":" << segment_index
               << ",\"attempt_index\":" << attempt_index << ",\"local_time\":";
      json_double(samples_, sample.first);
      samples_ << ",\"section\":" << section << ",\"phase\":[";
      for (size_t dof = 0; dof < dofs; ++dof)
      {
        if (dof) samples_ << ',';
        samples_ << phase_at(profiles[0].at(dof), sample.first);
      }
      samples_ << "],\"boundary_side\":" << json_string(sample.second) << ",\"q\":";
      json_double_vector(samples_, q, dofs);
      samples_ << ",\"dq\":";
      json_double_vector(samples_, dq, dofs);
      samples_ << ",\"ddq\":";
      json_double_vector(samples_, ddq, dofs);
      samples_ << ",\"native_jerk\":[";
      for (size_t dof = 0; dof < dofs; ++dof)
      {
        if (dof) samples_ << ',';
        json_double(samples_, jerk_at(profiles[0].at(dof), phase_at(profiles[0].at(dof), sample.first)));
      }
      samples_ << "]}\n";
    }
    samples_.flush();
  }

  void write_summary(const size_t calls, const size_t accepted, const size_t retries,
                     const std::vector<double>& final_durations, const std::vector<std::string>& sources,
                     const std::vector<bool>& extension_flags, const std::vector<bool>& final_assignments,
                     const size_t final_generation, const size_t last_waypoint_idx,
                     const bool smoothing_complete, const bool duration_ceiling_hit,
                     const ruckig::Result last_result)
  {
    if (!enabled_) return;
    certified_output_guard::assert_output_path_writable(directory_);
    calls_.flush();
    resolutions_.flush();
    overshoot_events_.flush();
    remediation_events_.flush();
    extension_calls_.flush();
    std::ofstream summary(directory_ / "stage25r_native_run_summary.json", std::ios::out | std::ios::trunc);
    summary << "{\"schema_version\":\"stage25r-native-run-summary-v1\",\"status\":\"completed\",\"calls_total\":"
            << calls << ",\"accepted_segment_calls\":" << accepted << ",\"retry_calls\":" << retries
            << ",\"overshoot_check_period\":";
    json_double(summary, kOvershootCheckPeriod);
    summary << ",\"final_segment_durations\":[";
    for (size_t i = 0; i < final_durations.size(); ++i) { if (i) summary << ','; json_double(summary, final_durations[i]); }
    summary << "],\"duration_sources\":[";
    for (size_t i = 0; i < sources.size(); ++i) { if (i) summary << ','; summary << json_string(sources[i]); }
    summary << "],\"changed_by_duration_extension\":[";
    for (size_t i = 0; i < extension_flags.size(); ++i) { if (i) summary << ','; summary << (extension_flags[i] ? "true" : "false"); }
    summary << "],\"changed_by_final_segment_assignment\":[";
    for (size_t i = 0; i < final_assignments.size(); ++i) { if (i) summary << ','; summary << (final_assignments[i] ? "true" : "false"); }
    summary << "],\"trajectory_generation_final\":" << final_generation
            << ",\"trajectory_generation_reset_count\":" << generation_reset_count_
            << ",\"last_called_waypoint_idx\":" << last_waypoint_idx
            << ",\"moveit_native_return_value\":" << (successful(last_result) ? "true" : "false")
            << ",\"moveit_last_ruckig_result\":{\"numeric_result\":" << static_cast<int>(last_result)
            << ",\"result_name\":" << json_string(result_name(last_result)) << "}"
            << ",\"moveit_smoothing_complete\":" << (smoothing_complete ? "true" : "false")
            << ",\"strict_stage25r2_completion\":" << (successful(last_result) && smoothing_complete ? "true" : "false")
            << ",\"duration_ceiling_hit\":" << (duration_ceiling_hit ? "true" : "false") << "}\n";
    summary.flush();

    // A single MoveIt process may smooth multiple independent primitives.
    // Preserve every invocation instead of retaining only the final summary.
    std::ofstream summaries(directory_ / "stage25r_native_run_summaries.jsonl", std::ios::out | std::ios::app);
    summaries << "{\"schema_version\":\"stage25r-native-run-summary-v2\",\"invocation_index\":"
              << run_summary_index_++ << ",\"status\":\"completed\",\"calls_total\":" << calls
              << ",\"accepted_segment_calls\":" << accepted << ",\"retry_calls\":" << retries
              << ",\"max_duration_extension_factor\":";
    json_double(summaries, kMaxDurationExtensionFactor);
    summaries << ",\"final_duration_extension_factor\":";
    // 1.1 is applied once per retry; recording the reconstructed terminal
    // factor makes the wrapper ceiling independently auditable.
    json_double(summaries, std::pow(kDurationExtensionFraction, static_cast<double>(retries)));
    summaries << ",\"last_called_waypoint_idx\":" << last_waypoint_idx
              << ",\"final_ruckig_result\":{\"numeric_result\":" << static_cast<int>(last_result)
              << ",\"result_name\":" << json_string(result_name(last_result)) << "}"
              << ",\"smoothing_complete\":" << (smoothing_complete ? "true" : "false")
              << ",\"duration_ceiling_hit\":" << (duration_ceiling_hit ? "true" : "false")
              << ",\"wrapper_returned_bool\":" << (successful(last_result) ? "true" : "false")
              << ",\"strict_completion\":"
              << (successful(last_result) && smoothing_complete && !duration_ceiling_hit ? "true" : "false")
              << "}\n";
    summaries.flush();

    std::ofstream stale(directory_ / "stage25r2_stale_accepted_calls.json", std::ios::out | std::ios::trunc);
    stale << "{\"schema_version\":\"stage25r2-stale-accepted-calls-v1\",\"count\":"
          << stale_accepted_call_indices_.size() << ",\"call_indices\":[";
    size_t index = 0;
    for (const size_t call_index : stale_accepted_call_indices_)
    {
      if (index++) stale << ',';
      stale << call_index;
    }
    stale << "],\"stale_evidence_used_for_final_certification\":false}\n";
    stale.flush();
  }

private:
  void write_model_bounds(std::ostream& out, const moveit::core::JointModelGroup* group)
  {
    const auto& names = group->getVariableNames();
    const auto& model = group->getParentModel();
    out << '[';
    for (size_t i = 0; i < names.size(); ++i)
    {
      if (i) out << ',';
      const auto& bounds = model.getVariableBounds(names.at(i));
      out << "{\"position_bounded\":" << (bounds.position_bounded_ ? "true" : "false")
          << ",\"min_position\":";
      json_double(out, bounds.min_position_);
      out << ",\"max_position\":";
      json_double(out, bounds.max_position_);
      out << ",\"velocity_bounded\":" << (bounds.velocity_bounded_ ? "true" : "false")
          << ",\"max_velocity\":";
      json_double(out, bounds.max_velocity_);
      out << ",\"acceleration_bounded\":" << (bounds.acceleration_bounded_ ? "true" : "false")
          << ",\"max_acceleration\":";
      json_double(out, bounds.max_acceleration_);
      out << ",\"jerk_bounded\":" << (bounds.jerk_bounded_ ? "true" : "false") << ",\"max_jerk\":";
      json_double(out, bounds.max_jerk_);
      out << ",\"source_joint_name\":" << json_string(names.at(i)) << "}";
    }
    out << ']';
  }

  void write_output(std::ostream& out, const DynamicTrajectory& output, const bool valid, const size_t dofs)
  {
    out << "{\"duration\":";
    json_double(out, output.get_duration());
    out << ",\"independent_min_durations\":";
    if (!valid) out << "null";
    else json_double_vector(out, output.get_independent_min_durations(), dofs);
    out << ",\"profile_available\":" << (valid ? "true" : "false");
    if (!valid)
    {
      out << ",\"profiles\":null,\"position_extrema\":null,\"kinematic_extrema\":null}";
      return;
    }
    auto profiles = output.get_profiles();
    if (compact_probe())
      out << ",\"profiles\":null";
    else
    {
      out << ",\"profiles\":[";
      for (size_t dof = 0; dof < dofs; ++dof)
      {
        if (dof) out << ',';
        const auto& p = profiles[0].at(dof);
        out << "{\"phase_duration\":[";
        for (size_t i = 0; i < p.t.size(); ++i) { if (i) out << ','; json_double(out, p.t[i]); }
        out << "],\"cumulative_phase_boundaries\":[";
        for (size_t i = 0; i < p.t_sum.size(); ++i) { if (i) out << ','; json_double(out, p.t_sum[i]); }
        out << "],\"p\":[";
        for (size_t i = 0; i < p.p.size(); ++i) { if (i) out << ','; json_double(out, p.p[i]); }
        out << "],\"v\":[";
        for (size_t i = 0; i < p.v.size(); ++i) { if (i) out << ','; json_double(out, p.v[i]); }
        out << "],\"a\":[";
        for (size_t i = 0; i < p.a.size(); ++i) { if (i) out << ','; json_double(out, p.a[i]); }
        out << "],\"j\":[";
        for (size_t i = 0; i < p.j.size(); ++i) { if (i) out << ','; json_double(out, p.j[i]); }
        out << "],\"limits\":" << static_cast<int>(p.limits) << ",\"direction\":" << static_cast<int>(p.direction)
            << ",\"jerk_signs\":" << static_cast<int>(p.jerk_signs) << ",\"brake\":";
        write_brake(out, p.brake);
        out << "}";
      }
      out << "]";
    }
    out << ",\"position_extrema\":[";
    auto ruckig_reported_extrema = const_cast<DynamicTrajectory&>(output).get_position_extrema();
    for (size_t dof = 0; dof < dofs; ++dof)
    {
      if (dof) out << ',';
      const auto extrema = exact_position_extrema(profiles[0].at(dof));
      out << "{\"min\":"; json_double(out, extrema.min); out << ",\"max\":"; json_double(out, extrema.max);
      out << ",\"t_min\":"; json_double(out, extrema.t_min); out << ",\"t_max\":"; json_double(out, extrema.t_max); out << '}';
    }
    out << "],\"ruckig_0_9_2_reported_position_extrema\":[";
    for (size_t dof = 0; dof < dofs; ++dof)
    {
      if (dof) out << ',';
      const auto& extrema = ruckig_reported_extrema.at(dof);
      out << "{\"min\":"; json_double(out, extrema.min); out << ",\"max\":"; json_double(out, extrema.max);
      out << ",\"t_min\":"; json_double(out, extrema.t_min); out << ",\"t_max\":"; json_double(out, extrema.t_max); out << '}';
    }
    out << "],\"position_extrema_method\":\"analytic_roots_from_native_Ruckig_profile_including_zero_jerk_constant_acceleration_phases\",\"kinematic_extrema\":[";
    for (size_t dof = 0; dof < dofs; ++dof)
    {
      if (dof) out << ',';
      const auto ext = kinematic_extrema(profiles[0].at(dof));
      out << "{\"max_abs_velocity\":"; json_double(out, ext.max_abs_velocity);
      out << ",\"max_abs_acceleration\":"; json_double(out, ext.max_abs_acceleration);
      out << ",\"max_abs_jerk\":"; json_double(out, ext.max_abs_jerk); out << '}';
    }
    out << "]}";
  }

  bool enabled_ { false };
  std::filesystem::path directory_;
  std::ofstream calls_;
  std::ofstream resolutions_;
  std::ofstream samples_;
  std::ofstream overshoot_events_;
  std::ofstream generation_events_;
  std::ofstream remediation_events_;
  std::ofstream extension_calls_;
  std::vector<size_t> accepted_call_indices_;
  std::set<size_t> stale_accepted_call_indices_;
  size_t generation_reset_count_ { 0 };
  size_t run_summary_index_ { 0 };
};

NativeRecorder& recorder()
{
  static NativeRecorder value;
  return value;
}

void entry_marker(const char* label)
{
  const char* raw_dir = std::getenv("STAGE25R_NATIVE_DIR");
  if (!raw_dir || !*raw_dir) return;
  certified_output_guard::assert_output_path_writable(std::filesystem::path(raw_dir));
  std::ofstream marker(std::filesystem::path(raw_dir) / (std::string("stage25r_entry_") + label + ".marker"),
                       std::ios::out | std::ios::app);
  marker << label << "\n";
}

void stage25r_initialize_ruckig_state(const moveit::core::RobotState& first_waypoint,
                                      const moveit::core::JointModelGroup* joint_group, DynamicInput& input)
{
  const size_t dofs = joint_group->getVariableCount();
  const auto& indices = joint_group->getVariableIndexList();
  for (size_t i = 0; i < dofs; ++i)
  {
    input.current_position.at(i) = first_waypoint.getVariablePosition(indices.at(i));
    input.current_velocity.at(i) = std::clamp(first_waypoint.getVariableVelocity(indices.at(i)),
                                              -input.max_velocity.at(i), input.max_velocity.at(i));
    input.current_acceleration.at(i) = std::clamp(first_waypoint.getVariableAcceleration(indices.at(i)),
                                                  -input.max_acceleration.at(i), input.max_acceleration.at(i));
  }
}

void stage25r_get_next_ruckig_input(const moveit::core::RobotStateConstPtr& current_waypoint,
                                    const moveit::core::RobotStateConstPtr& next_waypoint,
                                    const moveit::core::JointModelGroup* joint_group, DynamicInput& input)
{
  const size_t dofs = joint_group->getVariableCount();
  const auto& indices = joint_group->getVariableIndexList();
  for (size_t joint = 0; joint < dofs; ++joint)
  {
    input.current_position.at(joint) = current_waypoint->getVariablePosition(indices.at(joint));
    input.current_velocity.at(joint) = std::clamp(current_waypoint->getVariableVelocity(indices.at(joint)),
                                                  -input.max_velocity.at(joint), input.max_velocity.at(joint));
    input.current_acceleration.at(joint) = std::clamp(current_waypoint->getVariableAcceleration(indices.at(joint)),
                                                      -input.max_acceleration.at(joint), input.max_acceleration.at(joint));
    input.target_position.at(joint) = next_waypoint->getVariablePosition(indices.at(joint));
    input.target_velocity.at(joint) = std::clamp(next_waypoint->getVariableVelocity(indices.at(joint)),
                                                 -input.max_velocity.at(joint), input.max_velocity.at(joint));
    input.target_acceleration.at(joint) = std::clamp(next_waypoint->getVariableAcceleration(indices.at(joint)),
                                                     -input.max_acceleration.at(joint), input.max_acceleration.at(joint));
  }
}

void stage25r_extend_duration(const double factor, const size_t waypoint_idx, const size_t dofs,
                              const std::vector<int>& move_group_indices,
                              const robot_trajectory::RobotTrajectory& original,
                              robot_trajectory::RobotTrajectory& trajectory)
{
  trajectory.setWayPointDurationFromPrevious(waypoint_idx + 1,
                                             factor * original.getWayPointDurationFromPrevious(waypoint_idx + 1));
  auto target = trajectory.getWayPointPtr(waypoint_idx + 1);
  const auto previous = trajectory.getWayPointPtr(waypoint_idx);
  const double timestep = trajectory.getWayPointDurationFromPrevious(waypoint_idx + 1);
  for (size_t joint = 0; joint < dofs; ++joint)
  {
    const int index = move_group_indices.at(joint);
    target->setVariableVelocity(index, target->getVariableVelocity(index) / factor);
    const double previous_velocity = previous->getVariableVelocity(index);
    const double target_velocity = target->getVariableVelocity(index);
    target->setVariableAcceleration(index, (target_velocity - previous_velocity) / timestep);
  }
}

OvershootCheckResult stage25r_check_overshoot(DynamicTrajectory& output, const size_t dofs, const DynamicInput& input,
                                              const double threshold)
{
  OvershootCheckResult result;
  const auto profiles = output.get_profiles();
  for (double time_from_start = kOvershootCheckPeriod; time_from_start < output.get_duration();
       time_from_start += kOvershootCheckPeriod)
  {
    std::vector<double> position(dofs), velocity(dofs), acceleration(dofs);
    size_t section = 0;
    output.at_time(time_from_start, position, velocity, acceleration, section);
    for (size_t joint = 0; joint < dofs; ++joint)
    {
      const double current_minus_target = input.current_position.at(joint) - input.target_position.at(joint);
      const double sample_minus_target = position[joint] - input.target_position.at(joint);
      const double signed_ratio = sample_minus_target / current_minus_target;
      const double abs_overshoot = std::fabs(sample_minus_target);
      if ((signed_ratio < 0.0) && abs_overshoot > threshold)
      {
        OvershootPoint point;
        point.joint_index = joint;
        point.local_time = time_from_start;
        point.current_position = input.current_position.at(joint);
        point.target_position = input.target_position.at(joint);
        point.sampled_position = position[joint];
        point.current_minus_target = current_minus_target;
        point.sample_minus_target = sample_minus_target;
        point.signed_ratio = signed_ratio;
        point.abs_overshoot_rad = abs_overshoot;
        point.overshoot_threshold_rad = threshold;
        point.native_output_duration = output.get_duration();
        point.section = section;
        point.q = position;
        point.dq = velocity;
        point.ddq = acceleration;
        point.phase = phase_at(profiles[0].at(joint), time_from_start);
        point.native_jerk.resize(dofs, 0.0);
        for (size_t index = 0; index < dofs; ++index)
          point.native_jerk[index] = jerk_at(profiles[0].at(index), phase_at(profiles[0].at(index), time_from_start));
        result.overshoots = true;
        if (!result.first)
          result.first = point;
        if (!result.worst || point.abs_overshoot_rad > result.worst->abs_overshoot_rad)
          result.worst = point;
      }
    }
  }
  return result;
}

bool stage3_h12_r5_hard_limits_certified(DynamicTrajectory& output, const DynamicInput& input,
                                         const moveit::core::JointModelGroup* group)
{
  const size_t dofs = group->getVariableCount();
  const auto& names = group->getVariableNames();
  const auto& model = group->getParentModel();
  auto profiles = output.get_profiles();
  const auto within = [](const double value, const double limit) {
    const double tolerance = 1.0e-12 + 32.0 * std::numeric_limits<double>::epsilon() *
                                           std::max({ 1.0, std::abs(value), std::abs(limit) });
    return value <= limit + tolerance;
  };
  for (size_t joint = 0; joint < dofs; ++joint)
  {
    const auto& bounds = model.getVariableBounds(names.at(joint));
    const auto extrema = exact_position_extrema(profiles[0].at(joint));
    if (bounds.position_bounded_ &&
        (!within(extrema.max, bounds.max_position_) ||
         !within(-extrema.min, -bounds.min_position_)))
      return false;
    const auto kinetic = kinematic_extrema(profiles[0].at(joint));
    if (!within(kinetic.max_abs_velocity, input.max_velocity.at(joint)) ||
        !within(kinetic.max_abs_acceleration, input.max_acceleration.at(joint)) ||
        !within(kinetic.max_abs_jerk, input.max_jerk.at(joint)))
      return false;
  }
  return true;
}
}  // namespace

namespace trajectory_processing
{
// The Python binding calls this exported free function.  The distro shared
// library calls the private member directly from within the same DSO, so a
// member-only preload is not sufficient; interposing this public boundary
// preserves the exact RobotModel-bound setup and routes into the instrumented
// member loop below.
bool applyRuckigSmoothing(robot_trajectory::RobotTrajectory& trajectory, const double velocity_scaling_factor,
                          const double acceleration_scaling_factor, const bool mitigate_overshoot,
                          const double overshoot_threshold)
{
  entry_marker("free_function");
  const auto* group = trajectory.getGroup();
  if (!group) return false;
  ruckig::InputParameter<ruckig::DynamicDOFs> input(group->getVariableCount());
  entry_marker("before_model_bounds");
  constexpr double default_max_velocity = 5.0;
  constexpr double default_max_acceleration = 10.0;
  constexpr double default_max_jerk = 1000.0;
  const auto& names = group->getVariableNames();
  const auto& model = group->getParentModel();
  for (size_t i = 0; i < names.size(); ++i)
  {
    const auto& bounds = model.getVariableBounds(names.at(i));
    input.max_velocity.at(i) = velocity_scaling_factor * (bounds.velocity_bounded_ ? bounds.max_velocity_ : default_max_velocity);
    input.max_acceleration.at(i) = acceleration_scaling_factor * (bounds.acceleration_bounded_ ? bounds.max_acceleration_ : default_max_acceleration);
    input.max_jerk.at(i) = bounds.jerk_bounded_ ? bounds.max_jerk_ : default_max_jerk;
  }
  entry_marker("after_model_bounds");
  entry_marker("before_member_call");
  return RuckigSmoothing::runRuckig(trajectory, input, mitigate_overshoot, overshoot_threshold);
}

bool RuckigSmoothing::runRuckig(robot_trajectory::RobotTrajectory& trajectory,
                                ruckig::InputParameter<ruckig::DynamicDOFs>& ruckig_input,
                                const bool mitigate_overshoot, const double overshoot_threshold)
{
  entry_marker("member_function");
  const size_t num_waypoints = trajectory.getWayPointCount();
  const moveit::core::JointModelGroup* const group = trajectory.getGroup();
  const size_t num_dof = group->getVariableCount();
  run_h12_r5a_installed_helper_probe(trajectory, group);
  ruckig::Trajectory<ruckig::DynamicDOFs, ruckig::StandardVector> ruckig_output(num_dof);
  const bool hard_limit_certification_rule = env_flag("H12_R5_HARD_LIMIT_CERTIFICATION");
  const bool assign_all_native_durations = env_flag("H12_R5_ASSIGN_ALL_NATIVE_DURATIONS");
  const bool use_installed_moveit_helpers = env_flag("H12_R5_USE_INSTALLED_MOVEIT_HELPERS");
  const bool boundary_velocity_recondition = env_flag("H12_R5_BOUNDARY_VELOCITY_RECONDITION");
  entry_marker("after_output_construct");
  trajectory.unwind();
  entry_marker("after_unwind");
  ruckig::Ruckig<ruckig::DynamicDOFs> ruckig(num_dof, trajectory.getAverageSegmentDuration());
  entry_marker("after_ruckig_construct");
  if (use_installed_moveit_helpers)
    RuckigSmoothing::initializeRuckigState(*trajectory.getFirstWayPointPtr(), group, ruckig_input);
  else
    stage25r_initialize_ruckig_state(*trajectory.getFirstWayPointPtr(), group, ruckig_input);
  entry_marker("after_initialize");
  robot_trajectory::RobotTrajectory original_trajectory = robot_trajectory::RobotTrajectory(trajectory, true /* deep copy */);
  entry_marker("after_original_copy");

  std::vector<double> original_durations(num_waypoints, 0.0);
  std::vector<double> final_durations(num_waypoints, 0.0);
  std::vector<std::string> duration_sources(num_waypoints, "other");
  std::vector<bool> changed_by_extension(num_waypoints, false);
  std::vector<bool> changed_by_final_assignment(num_waypoints, false);
  for (size_t i = 1; i < num_waypoints; ++i)
  {
    original_durations[i] = original_trajectory.getWayPointDurationFromPrevious(i);
    final_durations[i] = original_durations[i];
    duration_sources[i] = "original_TOTG";
  }

  auto& native = recorder();
  ruckig::Result ruckig_result = ruckig::Result::Error;
  double duration_extension_factor = 1.0;
  bool smoothing_complete = false;
  size_t waypoint_idx = 0;
  size_t global_call_index = 0;
  size_t accepted_calls = 0;
  size_t retry_calls = 0;
  std::vector<size_t> attempts(num_waypoints, 0);
  size_t trajectory_generation = 0;
  bool duration_ceiling_hit = false;
  bool terminal_native_failure = false;

  while ((duration_extension_factor <= kMaxDurationExtensionFactor) && !smoothing_complete && !terminal_native_failure)
  {
    while (waypoint_idx < num_waypoints - 1)
    {
      moveit::core::RobotStatePtr curr_waypoint = trajectory.getWayPointPtr(waypoint_idx);
      moveit::core::RobotStatePtr next_waypoint = trajectory.getWayPointPtr(waypoint_idx + 1);
      if (use_installed_moveit_helpers)
        RuckigSmoothing::getNextRuckigInput(curr_waypoint, next_waypoint, group, ruckig_input);
      else
        stage25r_get_next_ruckig_input(curr_waypoint, next_waypoint, group, ruckig_input);

      const size_t call_index = global_call_index++;
      const size_t attempt_index = attempts.at(waypoint_idx)++;
      const double segment_duration_before = trajectory.getWayPointDurationFromPrevious(waypoint_idx + 1);

      if (boundary_velocity_recondition)
      {
        ruckig::Ruckig<ruckig::DynamicDOFs> validator(num_dof, 0.01);
        if (!validator.validate_input(ruckig_input, true, false))
        {
          const DynamicInput original_input = ruckig_input;
          bool repaired = false;
          const std::array<double, 7> factors{ 0.999, 0.995, 0.99, 0.95, 0.9, 0.5, 0.0 };
          for (const double factor : factors)
          {
            for (size_t joint = 0; joint < num_dof && !repaired; ++joint)
            {
              if (std::abs(original_input.current_velocity.at(joint)) < 0.9 * original_input.max_velocity.at(joint))
                continue;
              ruckig_input = original_input;
              ruckig_input.current_velocity.at(joint) *= factor;
              if (validator.validate_input(ruckig_input, true, false))
              {
                const int variable_index = group->getVariableIndexList().at(joint);
                curr_waypoint->setVariableVelocity(variable_index, ruckig_input.current_velocity.at(joint));
                native.write_remediation(call_index, waypoint_idx, joint,
                                         original_input.current_velocity.at(joint),
                                         ruckig_input.current_velocity.at(joint), factor, "current");
                repaired = true;
              }
            }
            if (repaired) break;
          }
          if (!repaired)
            ruckig_input = original_input;
        }
        if (!validator.validate_input(ruckig_input, false, true))
        {
          const DynamicInput original_input = ruckig_input;
          bool repaired = false;
          const std::array<double, 7> factors{ 0.999, 0.995, 0.99, 0.95, 0.9, 0.5, 0.0 };
          for (const double factor : factors)
          {
            for (size_t joint = 0; joint < num_dof && !repaired; ++joint)
            {
              if (std::abs(original_input.target_velocity.at(joint)) < 0.9 * original_input.max_velocity.at(joint))
                continue;
              ruckig_input = original_input;
              ruckig_input.target_velocity.at(joint) *= factor;
              if (validator.validate_input(ruckig_input, false, true))
              {
                const int variable_index = group->getVariableIndexList().at(joint);
                next_waypoint->setVariableVelocity(variable_index, ruckig_input.target_velocity.at(joint));
                native.write_remediation(call_index, waypoint_idx, joint,
                                         original_input.target_velocity.at(joint),
                                         ruckig_input.target_velocity.at(joint), factor, "target");
                repaired = true;
              }
            }
            if (repaired) break;
          }
          if (!repaired)
            ruckig_input = original_input;
        }
      }
      const DynamicInput input_before_calculate = ruckig_input;
      ruckig_result = ruckig.calculate(ruckig_input, ruckig_output);

      // This is intentionally the first operation after calculate().  The
      // native output/profile snapshot is written before overshoot handling.
      native.write_immediate(call_index, waypoint_idx, attempt_index, trajectory_generation, duration_extension_factor,
                             segment_duration_before, input_before_calculate, ruckig_input,
                             ruckig_output, ruckig_result, group);

      OvershootCheckResult overshoot_result;
      if (mitigate_overshoot)
      {
        overshoot_result = stage25r_check_overshoot(ruckig_output, num_dof, ruckig_input, overshoot_threshold);
        if (use_installed_moveit_helpers)
          overshoot_result.overshoots = RuckigSmoothing::checkOvershoot(
              ruckig_output, num_dof, ruckig_input, overshoot_threshold);
      }
      native.write_overshoot_event(call_index, waypoint_idx, attempt_index, trajectory_generation,
                                   duration_extension_factor, overshoot_result);

      // H12-R5 Case B rule: retain MoveIt's authoritative intermediate-target
      // overshoot observation, but classify it as benign only when exact native
      // position extrema and analytic velocity/acceleration/jerk extrema all
      // remain inside the installed RobotModel/Ruckig hard limits.  This is
      // opt-in and leaves the historical R4 behavior byte-for-byte selectable.
      const bool certified_benign_overshoot =
          hard_limit_certification_rule && overshoot_result.overshoots && successful(ruckig_result) &&
          stage3_h12_r5_hard_limits_certified(ruckig_output, ruckig_input, group);
      OvershootCheckResult effective_overshoot = overshoot_result;
      if (certified_benign_overshoot)
        effective_overshoot.overshoots = false;

      const bool call_success = successful(ruckig_result);
      if (!call_success)
      {
        native.write_resolution(call_index, waypoint_idx, attempt_index, trajectory_generation,
                                duration_extension_factor, overshoot_result, false, false,
                                trajectory.getWayPointDurationFromPrevious(waypoint_idx + 1),
                                "native_ruckig_error_no_duration_mutation", false, 0.0);
        terminal_native_failure = true;
        break;
      }
      const bool final_assignment = !effective_overshoot.overshoots && waypoint_idx == num_waypoints - 2 && call_success;
      if (final_assignment)
      {
        trajectory.setWayPointDurationFromPrevious(waypoint_idx + 1, ruckig_output.get_duration());
        final_durations[waypoint_idx + 1] = ruckig_output.get_duration();
        duration_sources[waypoint_idx + 1] = certified_benign_overshoot ?
            "native_ruckig_output_duration_hard_limit_certified" : "native_ruckig_output_duration";
        changed_by_final_assignment[waypoint_idx + 1] = true;
        ++accepted_calls;
        native.write_resolution(call_index, waypoint_idx, attempt_index, trajectory_generation, duration_extension_factor,
                                overshoot_result, true, false,
                                trajectory.getWayPointDurationFromPrevious(waypoint_idx + 1), duration_sources[waypoint_idx + 1],
                                true, ruckig_output.get_duration());
        native.write_samples(call_index, waypoint_idx, attempt_index, ruckig_output);
        smoothing_complete = true;
        break;
      }

      if (effective_overshoot.overshoots)
      {
        duration_extension_factor *= kDurationExtensionFraction;
        trajectory = robot_trajectory::RobotTrajectory(original_trajectory, true /* deep copy */);
        const size_t previous_generation = trajectory_generation;
        ++trajectory_generation;
        native.write_generation_reset(previous_generation, trajectory_generation, waypoint_idx, call_index);
        const std::vector<int>& move_group_idx = group->getVariableIndexList();
        if (use_installed_moveit_helpers)
        {
          const auto original_snapshot = capture_h12_r5a_helper_snapshot(original_trajectory, move_group_idx);
          const auto before_snapshot = capture_h12_r5a_helper_snapshot(trajectory, move_group_idx);
          const size_t second_argument_passed = waypoint_idx;
          RuckigSmoothing::extendTrajectoryDuration(duration_extension_factor, waypoint_idx, num_dof,
                                                     move_group_idx, original_trajectory, trajectory);
          const auto after_snapshot = capture_h12_r5a_helper_snapshot(trajectory, move_group_idx);
          const bool invariant_passed = h12_r5a_helper_mutation_invariant(
              before_snapshot, after_snapshot, original_snapshot, waypoint_idx, duration_extension_factor);
          native.write_extension_call(call_index, waypoint_idx, duration_extension_factor, second_argument_passed,
                                      original_snapshot, before_snapshot, after_snapshot, invariant_passed);
          if (!invariant_passed)
          {
            std::cerr << "H12-R5A installed helper mutation invariant failed; refusing to continue\n";
            return false;
          }
        }
        else
          stage25r_extend_duration(duration_extension_factor, waypoint_idx, num_dof, move_group_idx,
                                   original_trajectory, trajectory);
        final_durations[waypoint_idx + 1] = trajectory.getWayPointDurationFromPrevious(waypoint_idx + 1);
        duration_sources[waypoint_idx + 1] = "duration_extension_retry";
        changed_by_extension[waypoint_idx + 1] = true;
        ++retry_calls;
        native.write_resolution(call_index, waypoint_idx, attempt_index, previous_generation, duration_extension_factor,
                                overshoot_result, false, true,
                                final_durations[waypoint_idx + 1], duration_sources[waypoint_idx + 1], false,
                                ruckig_output.get_duration());
        if (use_installed_moveit_helpers)
          RuckigSmoothing::initializeRuckigState(*trajectory.getFirstWayPointPtr(), group, ruckig_input);
        else
          stage25r_initialize_ruckig_state(*trajectory.getFirstWayPointPtr(), group, ruckig_input);
        break;
      }

      // MoveIt consumes Ruckig here as a feasibility/overshoot smoother.  It
      // does not replace every pre-existing TOTG segment duration with the
      // per-segment Ruckig minimum duration; only the final accepted segment is
      // assigned above.  Replacing every duration changes the Hermite path and
      // produced a 286x trajectory-duration inflation in the first R5 probe.
      if (assign_all_native_durations)
      {
        trajectory.setWayPointDurationFromPrevious(waypoint_idx + 1, ruckig_output.get_duration());
        final_durations[waypoint_idx + 1] = ruckig_output.get_duration();
        duration_sources[waypoint_idx + 1] = certified_benign_overshoot ?
            "native_ruckig_output_duration_hard_limit_certified" : "native_ruckig_output_duration";
        changed_by_final_assignment[waypoint_idx + 1] = true;
      }
      ++accepted_calls;
      native.write_resolution(call_index, waypoint_idx, attempt_index, trajectory_generation, duration_extension_factor,
                              overshoot_result, true, false,
                              trajectory.getWayPointDurationFromPrevious(waypoint_idx + 1), duration_sources[waypoint_idx + 1],
                              false, ruckig_output.get_duration());
      native.write_samples(call_index, waypoint_idx, attempt_index, ruckig_output);
      ++waypoint_idx;
    }
  }

  duration_ceiling_hit = duration_extension_factor > kMaxDurationExtensionFactor && !smoothing_complete;
  if (duration_ceiling_hit)
  {
    // Preserve the observable MoveIt diagnostic while retaining exact native
    // evidence for the fail-closed H7.4 completion checker.
    std::cerr << "Ruckig extended the trajectory duration to its maximum and still did not find a solution\n";
  }

  for (size_t i = 1; i < num_waypoints; ++i)
  {
    final_durations[i] = trajectory.getWayPointDurationFromPrevious(i);
  }
  native.write_summary(global_call_index, accepted_calls, retry_calls, final_durations, duration_sources,
                       changed_by_extension, changed_by_final_assignment, trajectory_generation,
                       waypoint_idx, smoothing_complete, duration_ceiling_hit, ruckig_result);

  // Match MoveIt 2.12.4 exactly: after the extension loop it reports the
  // maximum-extension diagnostic, but returns based on the last Ruckig
  // Result only.  Do not add a smoothing_complete gate here.
  if (!successful(ruckig_result))
  {
    return false;
  }
  return true;
}
}  // namespace trajectory_processing
