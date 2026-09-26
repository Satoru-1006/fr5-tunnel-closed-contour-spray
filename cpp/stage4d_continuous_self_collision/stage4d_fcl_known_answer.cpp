#include <cmath>
#include <iostream>
#include <memory>

#include <fcl/geometry/shape/box.h>
#include <fcl/narrowphase/continuous_collision.h>

namespace {

bool run_case(const fcl::CollisionGeometryd* first,
              const fcl::Transform3d& first_begin,
              const fcl::Transform3d& first_end,
              const fcl::CollisionGeometryd* second,
              const fcl::Transform3d& second_begin,
              const fcl::Transform3d& second_end,
              bool expected_collision,
              const char* name)
{
  fcl::ContinuousCollisionRequestd request(
      100, 1.0e-9, fcl::CCDM_SCREW, fcl::GST_LIBCCD, fcl::CCDC_CONSERVATIVE_ADVANCEMENT);
  fcl::ContinuousCollisionResultd result;
  const double continuous_return = fcl::continuousCollide(
      first, first_begin, first_end, second, second_begin, second_end, request, result);
  const bool pass = continuous_return >= 0.0 && result.is_collide == expected_collision;
  std::cout << "{\"case\":\"" << name << "\",\"continuous_return\":" << continuous_return
            << ",\"collision\":" << (result.is_collide ? "true" : "false")
            << ",\"expected_collision\":" << (expected_collision ? "true" : "false")
            << ",\"time_of_contact\":" << (result.is_collide ? std::to_string(result.time_of_contact) : "null")
            << ",\"api_status\":\"" << (continuous_return >= 0.0 ? "OK" : "ERROR")
            << "\",\"status\":\"" << (pass ? "PASS" : "FAIL") << "\"}\n";
  return pass;
}

}  // namespace

int main()
{
  auto box = std::make_shared<fcl::Boxd>(0.2, 0.2, 0.2);
  fcl::Transform3d first_begin = fcl::Transform3d::Identity();
  fcl::Transform3d first_end = fcl::Transform3d::Identity();
  fcl::Transform3d second_begin = fcl::Transform3d::Identity();
  fcl::Transform3d second_end = fcl::Transform3d::Identity();
  first_begin.translation() = fcl::Vector3d(-1.0, 0.0, 0.0);
  first_end.translation() = fcl::Vector3d(1.0, 0.0, 0.0);
  const bool crossing = run_case(box.get(), first_begin, first_end, box.get(),
                                 second_begin, second_end, true, "forced_crossing");
  second_begin.translation() = fcl::Vector3d(0.0, 1.0, 0.0);
  second_end = second_begin;
  const bool separated = run_case(box.get(), first_begin, first_end, box.get(),
                                  second_begin, second_end, false, "known_separation");
  return crossing && separated ? 0 : 1;
}
