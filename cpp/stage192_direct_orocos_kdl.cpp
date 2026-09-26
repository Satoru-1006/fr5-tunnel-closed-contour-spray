#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <urdf/model.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
std::vector<double> read_f64(const std::string& path, std::size_t count) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<double> values(count);
  in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(double)));
  if (in.gcount() != static_cast<std::streamsize>(count * sizeof(double))) throw std::runtime_error("short binary input " + path);
  return values;
}

KDL::Frame target_frame(const std::vector<double>& m) {
  KDL::Rotation r(m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]);
  return {r, {m[3], m[7], m[11]}};
}

double position_error(const KDL::Frame& actual, const KDL::Frame& target) {
  return (actual.p - target.p).Norm();
}

std::string json_array(const std::vector<double>& values) {
  std::ostringstream out; out << "[" << std::setprecision(17);
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ","; out << values[i]; }
  out << "]"; return out.str();
}

struct Args { std::string urdf, target, seed, limits, output, mode = "same_instance"; int repeat = 100; };

Args parse(int argc, char** argv) {
  Args args;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string key(argv[i]), value(argv[i + 1]);
    if (key == "--urdf") args.urdf = value; else if (key == "--target") args.target = value; else if (key == "--seed") args.seed = value;
    else if (key == "--limits") args.limits = value; else if (key == "--output") args.output = value; else if (key == "--mode") args.mode = value; else if (key == "--repeat") args.repeat = std::stoi(value);
  }
  if (args.urdf.empty() || args.target.empty() || args.seed.empty() || args.limits.empty() || args.output.empty()) throw std::runtime_error("missing direct KDL arguments");
  return args;
}
}

int main(int argc, char** argv) {
  try {
    const Args args = parse(argc, argv);
    const auto target = target_frame(read_f64(args.target, 16));
    const auto seed = read_f64(args.seed, 6);
    const auto limits = read_f64(args.limits, 12);
    urdf::Model urdf;
    if (!urdf.initFile(args.urdf)) throw std::runtime_error("URDF parse failed");
    KDL::Tree tree;
    if (!kdl_parser::treeFromUrdfModel(urdf, tree)) throw std::runtime_error("KDL tree parse failed");
    KDL::Chain chain;
    if (!tree.getChain("base_link", "spray_tcp_link", chain)) throw std::runtime_error("KDL chain extraction failed");
    KDL::JntArray qmin(6), qmax(6);
    for (int i = 0; i < 6; ++i) { qmin(i) = limits[2 * i]; qmax(i) = limits[2 * i + 1]; }
    std::ofstream out(args.output);
    out << "{\"schema_version\":\"1.0\",\"backend\":\"direct_orocos_kdl\",\"mode\":\"" << args.mode << "\",\"repeat\":" << args.repeat << ",\"records\":[";
    bool first = true;
    KDL::ChainIkSolverPos_NR_JL* shared = nullptr;
    KDL::ChainFkSolverPos_recursive* shared_fk = nullptr;
    KDL::ChainIkSolverVel_pinv* shared_vel = nullptr;
    if (args.mode == "same_instance") {
      shared_fk = new KDL::ChainFkSolverPos_recursive(chain);
      shared_vel = new KDL::ChainIkSolverVel_pinv(chain, 1e-5, 150);
      shared = new KDL::ChainIkSolverPos_NR_JL(chain, qmin, qmax, *shared_fk, *shared_vel, 500, 1e-5);
    }
    for (int iteration = 0; iteration < args.repeat; ++iteration) {
      KDL::ChainFkSolverPos_recursive local_fk(chain);
      KDL::ChainIkSolverVel_pinv local_vel(chain, 1e-5, 150);
      KDL::ChainIkSolverPos_NR_JL local(chain, qmin, qmax, local_fk, local_vel, 500, 1e-5);
      KDL::ChainIkSolverPos_NR_JL& solver = shared ? *shared : local;
      KDL::JntArray qseed(6), solution(6);
      for (int i = 0; i < 6; ++i) qseed(i) = seed[i];
      const int code = solver.CartToJnt(qseed, target, solution);
      std::vector<double> q(6); for (int i = 0; i < 6; ++i) q[i] = solution(i);
      KDL::Frame actual; double fk_error = -1.0;
      if (code >= 0) { local_fk.JntToCart(solution, actual); fk_error = position_error(actual, target); }
      if (!first) out << ","; first = false;
      out << "{\"iteration\":" << iteration << ",\"process_id\":" << static_cast<long long>(::getpid()) << ",\"solver_instance_id\":\"direct_kdl:" << args.mode << ":" << (args.mode == "same_instance" ? 1 : iteration + 1) << "\",\"direct_kdl_success\":" << (code >= 0 ? "true" : "false") << ",\"direct_kdl_error_code\":" << code << ",\"direct_kdl_solution\":" << json_array(q) << ",\"direct_kdl_fk_error\":" << std::setprecision(17) << fk_error << "}";
    }
    out << "]}\n";
    delete shared; delete shared_vel; delete shared_fk;
    return 0;
  } catch (const std::exception& error) { std::cerr << "stage192_direct_orocos_kdl: " << error.what() << "\n"; return 2; }
}
