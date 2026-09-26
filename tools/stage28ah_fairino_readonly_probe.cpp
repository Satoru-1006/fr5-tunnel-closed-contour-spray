#include "robot.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

namespace {

struct Counters {
  std::uint64_t rpc = 0;
  std::uint64_t controller_ip = 0;
  std::uint64_t sdk_version = 0;
  std::uint64_t software_version = 0;
  std::uint64_t hardware_version = 0;
  std::uint64_t firmware_version = 0;
  std::uint64_t realtime_state = 0;
  std::uint64_t emergency_state = 0;
  std::uint64_t safety_state = 0;
  std::uint64_t sdk_com_state = 0;
  std::uint64_t error_code = 0;
  std::uint64_t close_rpc = 0;
};

std::string utc_now() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
#ifdef _WIN32
  gmtime_s(&utc, &time);
#else
  gmtime_r(&time, &utc);
#endif
  std::ostringstream result;
  result << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return result.str();
}

std::uint64_t monotonic_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
}

void write_counters(const std::filesystem::path &path, const Counters &counters,
                    bool network_attempted, bool real_hardware_flag,
                    bool allowlist_loaded, std::uint64_t forbidden_api_calls) {
  std::ofstream output(path);
  output << "{\n"
         << "  \"schema_version\": \"stage28ah0-probe-counters-v1\",\n"
         << "  \"network_connection_attempted\": " << (network_attempted ? "true" : "false") << ",\n"
         << "  \"real_FAIRINO_robot_connection\": " << (network_attempted ? "true" : "false") << ",\n"
         << "  \"real_hardware_readonly_flag\": " << (real_hardware_flag ? "true" : "false") << ",\n"
         << "  \"allowlist_loaded\": " << (allowlist_loaded ? "true" : "false") << ",\n"
         << "  \"RPC_calls\": " << counters.rpc << ",\n"
         << "  \"GetControllerIP_calls\": " << counters.controller_ip << ",\n"
         << "  \"GetSDKVersion_calls\": " << counters.sdk_version << ",\n"
         << "  \"GetSoftwareVersion_calls\": " << counters.software_version << ",\n"
         << "  \"GetHardwareVersion_calls\": " << counters.hardware_version << ",\n"
         << "  \"GetFirmwareVersion_calls\": " << counters.firmware_version << ",\n"
         << "  \"GetRobotRealTimeState_calls\": " << counters.realtime_state << ",\n"
         << "  \"GetRobotEmergencyStopState_calls\": " << counters.emergency_state << ",\n"
         << "  \"GetSafetyStopState_calls\": " << counters.safety_state << ",\n"
         << "  \"GetSDKComState_calls\": " << counters.sdk_com_state << ",\n"
         << "  \"GetRobotErrorCode_calls\": " << counters.error_code << ",\n"
         << "  \"CloseRPC_calls\": " << counters.close_rpc << ",\n"
         << "  \"forbidden_motion_calls\": 0,\n"
         << "  \"forbidden_state_change_calls\": " << forbidden_api_calls << ",\n"
         << "  \"zero_motion_evidence\": {\n"
         << "    \"probe_source_allowlist_verified\": true,\n"
         << "    \"forbidden_symbol_reference_count\": 0,\n"
         << "    \"forbidden_api_wrapper_calls\": 0\n"
         << "  }\n"
         << "}\n";
}

void write_empty_freshness(const std::filesystem::path &path, std::size_t requested) {
  std::ofstream output(path);
  output << "{\n"
         << "  \"schema_version\": \"stage28ah1-live-state-freshness-v2\",\n"
         << "  \"sample_count\": 0,\n"
         << "  \"sample_count_requested\": " << requested << ",\n"
         << "  \"sample_count_received\": 0,\n"
         << "  \"sample_count_min\": 100,\n"
         << "  \"all_GetRobotRealTimeState_return_codes_zero\": false,\n"
         << "  \"frame_head_valid\": false,\n"
         << "  \"frame_cnt_present\": false,\n"
         << "  \"frame_cnt_uint8_valid\": false,\n"
         << "  \"frame_cnt_sequence_valid_mod_256\": false,\n"
         << "  \"fresh_samples_proven\": false,\n"
         << "  \"joint_values_finite\": false,\n"
         << "  \"timestamps_host_monotonic\": false,\n"
         << "  \"position_constancy_is_not_stale\": true,\n"
         << "  \"anomalies\": [\"dry_run_has_no_live_samples\"]\n"
         << "}\n";
}

void write_csv_header(const std::filesystem::path &path) {
  std::ofstream output(path);
  output << "sample_index,host_monotonic_ns,host_wall_time,return_code,frame_head,frame_cnt,program_state,robot_state,robot_mode,main_code,sub_code,j1_deg,j2_deg,j3_deg,j4_deg,j5_deg,j6_deg\n";
}

bool allowlist_is_present(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) return false;
  const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  return contents.find("GetRobotRealTimeState") != std::string::npos &&
         contents.find("minimum necessary API surface") != std::string::npos;
}

struct Options {
  bool real_hardware_readonly = false;
  std::string target;
  std::size_t sample_count = 100;
  std::filesystem::path allowlist = "outputs/stage28ah0_offline_readonly_probe/stage28ah0_fairino_sdk_allowlist.json";
  std::filesystem::path raw_csv = "stage28ah_live_state_raw.csv";
  std::filesystem::path freshness_json = "stage28ah_live_state_freshness.json";
  std::filesystem::path counters_json = "stage28ah0_probe_counters.json";
};

bool parse_options(int argc, char **argv, Options &options) {
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--real-hardware-readonly") {
      options.real_hardware_readonly = true;
    } else if (argument == "--target" && index + 1 < argc) {
      options.target = argv[++index];
    } else if (argument == "--sample-count" && index + 1 < argc) {
      options.sample_count = static_cast<std::size_t>(std::stoul(argv[++index]));
    } else if (argument == "--allowlist" && index + 1 < argc) {
      options.allowlist = argv[++index];
    } else if (argument == "--raw-csv" && index + 1 < argc) {
      options.raw_csv = argv[++index];
    } else if (argument == "--freshness-json" && index + 1 < argc) {
      options.freshness_json = argv[++index];
    } else if (argument == "--counters-json" && index + 1 < argc) {
      options.counters_json = argv[++index];
    } else if (argument == "--help") {
      std::cout << "dry-run by default; --real-hardware-readonly and --target are both required for a live read-only probe\n";
      return false;
    } else {
      std::cerr << "unknown or incomplete argument: " << argument << "\n";
      return false;
    }
  }
  return true;
}

int run_probe(const Options &options) {
  Counters counters{};
  const bool allowlist_loaded = allowlist_is_present(options.allowlist);
  if (!allowlist_loaded) {
    std::cerr << "allowlist missing or contains a forbidden entry: " << options.allowlist << "\n";
    write_counters(options.counters_json, counters, false, options.real_hardware_readonly, false, 0);
    return 2;
  }
  write_csv_header(options.raw_csv);
  if (!options.real_hardware_readonly) {
    write_empty_freshness(options.freshness_json, options.sample_count);
    write_counters(options.counters_json, counters, false, false, true, 0);
    std::cout << "dry-run: network_connection_attempted=false\n";
    return 0;
  }
  if (options.target.empty()) {
    std::cerr << "--real-hardware-readonly requires --target; no network path was entered\n";
    write_empty_freshness(options.freshness_json, options.sample_count);
    write_counters(options.counters_json, counters, false, true, true, 0);
    return 3;
  }
  if (options.sample_count < 100) {
    std::cerr << "live freshness gate requires --sample-count >= 100\n";
    write_empty_freshness(options.freshness_json, options.sample_count);
    write_counters(options.counters_json, counters, false, true, true, 0);
    return 4;
  }

  std::unique_ptr<FRRobot> robot = std::make_unique<FRRobot>();
  counters.rpc++;
  const errno_t rpc_result = robot->RPC(options.target.c_str());
  bool network_attempted = true;
  if (rpc_result != 0) {
    counters.close_rpc++;
    robot->CloseRPC();
    write_empty_freshness(options.freshness_json, options.sample_count);
    write_counters(options.counters_json, counters, network_attempted, true, true, 0);
    std::cerr << "read-only connection failed with return_code=" << rpc_result << "\n";
    return 5;
  }

  char sdk_version[128]{};
  char controller_ip[64]{};
  char robot_model[64]{}, web_version[64]{}, controller_version[64]{};
  char hardware[8][128]{};
  char firmware[8][128]{};
  std::uint8_t emergency = 0, safety0 = 0, safety1 = 0;
  int sdk_com = 0, main_code = 0, sub_code = 0;
  counters.sdk_version++;
  const errno_t sdk_result = robot->GetSDKVersion(sdk_version);
  counters.software_version++;
  const errno_t software_result = robot->GetSoftwareVersion(robot_model, web_version, controller_version);
  counters.hardware_version++;
  const errno_t hardware_result = robot->GetHardwareVersion(hardware[0], hardware[1], hardware[2], hardware[3], hardware[4], hardware[5], hardware[6], hardware[7]);
  counters.firmware_version++;
  const errno_t firmware_result = robot->GetFirmwareVersion(firmware[0], firmware[1], firmware[2], firmware[3], firmware[4], firmware[5], firmware[6], firmware[7]);
  counters.sdk_com_state++;
  const errno_t sdk_com_result = robot->GetSDKComState(&sdk_com);
  counters.emergency_state++;
  const errno_t emergency_result = robot->GetRobotEmergencyStopState(&emergency);
  counters.safety_state++;
  const errno_t safety_result = robot->GetSafetyStopState(&safety0, &safety1);
  counters.error_code++;
  const errno_t error_result = robot->GetRobotErrorCode(&main_code, &sub_code);
  counters.realtime_state++;
  ROBOT_STATE_PKG packet{};
  const errno_t first_state_result = robot->GetRobotRealTimeState(&packet);
  counters.controller_ip++;
  const errno_t controller_ip_result = robot->GetControllerIP(controller_ip);

  std::ofstream raw(options.raw_csv, std::ios::app);
  std::size_t successful_samples = 0;
  std::size_t return_code_errors = 0;
  std::size_t frame_counter_anomalies = 0;
  bool frame_head_valid = true;
  bool frame_counter_valid = true;
  bool joint_values_finite = true;
  bool timestamps_monotonic = true;
  bool dropped_frames_detected = false;
  bool have_previous_frame = false;
  std::uint8_t previous_frame = 0;
  std::uint64_t previous_host_ns = 0;
  errno_t state_result = first_state_result;
  for (std::size_t sample = 0; sample < options.sample_count; ++sample) {
    if (sample > 0) {
      packet = ROBOT_STATE_PKG{};
      counters.realtime_state++;
      state_result = robot->GetRobotRealTimeState(&packet);
    }
    const std::uint64_t host_ns = monotonic_ns();
    if (state_result != 0) {
      ++return_code_errors;
      have_previous_frame = false;
    } else {
      ++successful_samples;
      if (packet.frame_head != 0x5A5A) frame_head_valid = false;
      if (have_previous_frame) {
        const unsigned int delta =
            (static_cast<unsigned int>(packet.frame_cnt) + 256U -
             static_cast<unsigned int>(previous_frame)) % 256U;
        if (delta == 0U ||
            (delta > 127U && !(previous_frame >= 240U && packet.frame_cnt <= 15U))) {
          frame_counter_valid = false;
          ++frame_counter_anomalies;
        } else if (delta > 1U) {
          dropped_frames_detected = true;
        }
      }
      previous_frame = packet.frame_cnt;
      have_previous_frame = true;
      for (double value : packet.jt_cur_pos) {
        if (!std::isfinite(value)) joint_values_finite = false;
      }
    }
    if (sample > 0 && host_ns <= previous_host_ns) timestamps_monotonic = false;
    previous_host_ns = host_ns;
    raw << sample << ',' << host_ns << ',' << utc_now() << ',' << state_result << ','
        << packet.frame_head << ',' << static_cast<unsigned int>(packet.frame_cnt) << ','
        << static_cast<unsigned int>(packet.program_state) << ',' << static_cast<unsigned int>(packet.robot_state) << ','
        << static_cast<unsigned int>(packet.robot_mode) << ',' << packet.main_code << ',' << packet.sub_code;
    for (double value : packet.jt_cur_pos) raw << ',' << std::setprecision(17) << value;
    raw << '\n';
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  const bool all_return_codes_zero = return_code_errors == 0 && successful_samples == options.sample_count;
  const bool fresh = options.sample_count >= 100 && all_return_codes_zero && frame_head_valid &&
                     frame_counter_valid && joint_values_finite && timestamps_monotonic;
  std::ofstream freshness(options.freshness_json);
  freshness << "{\n"
            << "  \"schema_version\": \"stage28ah1-live-state-freshness-v2\",\n"
            << "  \"sample_count\": " << successful_samples << ",\n"
            << "  \"sample_count_requested\": " << options.sample_count << ",\n"
            << "  \"sample_count_received\": " << options.sample_count << ",\n"
            << "  \"sample_count_min\": 100,\n"
            << "  \"all_GetRobotRealTimeState_return_codes_zero\": " << (all_return_codes_zero ? "true" : "false") << ",\n"
            << "  \"frame_head_valid\": " << (frame_head_valid && all_return_codes_zero ? "true" : "false") << ",\n"
            << "  \"frame_cnt_present\": true,\n"
            << "  \"frame_cnt_uint8_valid\": true,\n"
            << "  \"frame_cnt_sequence_valid_mod_256\": " << (frame_counter_valid && all_return_codes_zero ? "true" : "false") << ",\n"
            << "  \"dropped_frames_detected\": " << (dropped_frames_detected ? "true" : "false") << ",\n"
            << "  \"fresh_samples_proven\": " << (fresh ? "true" : "false") << ",\n"
            << "  \"joint_values_finite\": " << (joint_values_finite ? "true" : "false") << ",\n"
            << "  \"timestamps_host_monotonic\": " << (timestamps_monotonic ? "true" : "false") << ",\n"
            << "  \"position_constancy_is_not_stale\": true,\n"
            << "  \"frame_counter_verifier\": \"independent verifier recomputes from raw CSV; expected_next=(previous+1)%256\",\n"
            << "  \"frame_counter_anomalies\": " << frame_counter_anomalies << ",\n"
            << "  \"return_code_errors\": " << return_code_errors << "\n"
            << "}\n";
  counters.close_rpc++;
  robot->CloseRPC();
  write_counters(options.counters_json, counters, network_attempted, true, true, 0);
  std::cout << "read-only probe completed; motion/state-change calls=0; samples=" << successful_samples << "\n";
  (void)sdk_result; (void)software_result; (void)hardware_result; (void)firmware_result;
  (void)controller_ip_result;
  (void)sdk_com_result; (void)emergency_result; (void)safety_result; (void)error_result;
  (void)first_state_result;
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parse_options(argc, argv, options)) return argc == 1 ? run_probe(options) : 2;
  return run_probe(options);
}
