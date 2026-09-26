// Stage 2.7S feasibility probe for one bounded Ruckig trajectory containing
// the frozen Stage 2.5 pre-timing positions as intermediate positions.

#include <ruckig/ruckig.hpp>

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr std::size_t D = 6;
const std::array<double, D> V{0.4725,0.4725,0.4725,0.48,0.48,0.48};
const std::array<double, D> A{0.105,0.105,0.105,0.105,0.105,0.105};
const std::array<double, D> J{8,8,8,8,8,8};
const std::array<double, D> L{-3.0543,-4.6251,-2.8274,-4.6251,-3.0543,-3.0543};
const std::array<double, D> U{3.0543,1.4835,2.8274,1.4835,3.0543,3.0543};

std::vector<std::string> fields(const std::string& line) {
  std::vector<std::string> result; std::stringstream stream(line); std::string value;
  while (std::getline(stream, value, ',')) result.push_back(value);
  return result;
}
std::vector<std::array<double,D>> read_q(const std::string& path) {
  std::ifstream input(path); if (!input) throw std::runtime_error("cannot open CSV");
  std::string line; std::getline(input,line); auto header=fields(line); std::array<std::size_t,D> qidx{};
  for (auto& value : header) if (!value.empty() && value.back() == '\r') value.pop_back();
  for (std::size_t j=0;j<D;++j) { const std::string name="q"+std::to_string(j+1); bool found=false; for(std::size_t i=0;i<header.size();++i) if(header[i]==name){qidx[j]=i;found=true;} if(!found) throw std::runtime_error("missing q column "+name); }
  std::vector<std::array<double,D>> rows;
  while(std::getline(input,line)){ if(line.empty()) continue; auto f=fields(line); std::array<double,D> q{}; for(std::size_t j=0;j<D;++j) q[j]=std::stod(f[qidx[j]]); rows.push_back(q); }
  return rows;
}
std::string name(ruckig::Result r){switch(r){case ruckig::Result::Working:return"Working";case ruckig::Result::Finished:return"Finished";case ruckig::Result::Error:return"Error";case ruckig::Result::ErrorInvalidInput:return"ErrorInvalidInput";case ruckig::Result::ErrorTrajectoryDuration:return"ErrorTrajectoryDuration";case ruckig::Result::ErrorPositionalLimits:return"ErrorPositionalLimits";case ruckig::Result::ErrorExecutionTimeCalculation:return"ErrorExecutionTimeCalculation";case ruckig::Result::ErrorSynchronizationCalculation:return"ErrorSynchronizationCalculation";}return"unknown";}
}

int main(int argc,char**argv){
  if(argc!=2){std::cerr<<"usage: stage27s_global_ruckig_probe <pre_timing.csv>\n";return 2;}
  try{
    auto rows=read_q(argv[1]); if(rows.size()<2) throw std::runtime_error("too few rows");
    ruckig::Ruckig<ruckig::DynamicDOFs> ruckig(D,0.01);
    ruckig::InputParameter<ruckig::DynamicDOFs> input(D);
    input.current_position=std::vector<double>(rows.front().begin(),rows.front().end());
    input.target_position=std::vector<double>(rows.back().begin(),rows.back().end());
    input.intermediate_positions.reserve(rows.size()-2);
    for(std::size_t i=1;i+1<rows.size();++i) input.intermediate_positions.emplace_back(rows[i].begin(),rows[i].end());
    input.current_velocity=std::vector<double>(D,0.0); input.target_velocity=std::vector<double>(D,0.0);
    input.current_acceleration=std::vector<double>(D,0.0); input.target_acceleration=std::vector<double>(D,0.0);
    input.max_velocity=std::vector<double>(V.begin(),V.end()); input.max_acceleration=std::vector<double>(A.begin(),A.end()); input.max_jerk=std::vector<double>(J.begin(),J.end());
    input.min_position=std::vector<double>(L.begin(),L.end()); input.max_position=std::vector<double>(U.begin(),U.end());
    ruckig::Trajectory<ruckig::DynamicDOFs,ruckig::StandardVector> trajectory(D);
    const auto result=ruckig.calculate(input,trajectory);
    std::cout<<"rows="<<rows.size()<<" intermediates="<<input.intermediate_positions.size()<<" result="<<name(result)<<" numeric="<<static_cast<int>(result);
    if(result==ruckig::Result::Working||result==ruckig::Result::Finished){std::cout<<" duration="<<std::setprecision(17)<<trajectory.get_duration()<<" sections="<<trajectory.get_profiles().size()<<"\n";}else std::cout<<"\n";
    return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 3;}
}
