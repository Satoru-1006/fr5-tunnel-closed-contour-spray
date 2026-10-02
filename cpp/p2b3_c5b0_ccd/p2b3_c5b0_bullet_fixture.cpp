#include <btBulletCollisionCommon.h>

#include <cstdlib>
#include <iostream>

namespace {

struct SweepResult : btCollisionWorld::ConvexResultCallback {
  bool hit{false};
  btScalar addSingleResult(btCollisionWorld::LocalConvexResult &result, bool) override {
    if (result.m_hitFraction <= btScalar(1.0)) hit = true;
    return result.m_hitFraction;
  }
};

bool sweep_crosses_static_box(bool should_cross) {
  btDefaultCollisionConfiguration configuration;
  btCollisionDispatcher dispatcher(&configuration);
  btDbvtBroadphase broadphase;
  btCollisionWorld world(&dispatcher, &broadphase, &configuration);

  btBoxShape obstacle_shape(btVector3(btScalar(0.1), btScalar(0.5), btScalar(0.5)));
  btCollisionObject obstacle;
  obstacle.setCollisionShape(&obstacle_shape);
  obstacle.setWorldTransform(btTransform(btQuaternion::getIdentity(), btVector3(0, 0, 0)));
  world.addCollisionObject(&obstacle);

  btSphereShape moving_shape(btScalar(0.1));
  btTransform from(btQuaternion::getIdentity(), btVector3(should_cross ? -1 : -1, 0, 0));
  btTransform to(btQuaternion::getIdentity(), btVector3(should_cross ? 1 : -0.4, 0, 0));
  SweepResult result;
  world.convexSweepTest(&moving_shape, from, to, result);
  world.removeCollisionObject(&obstacle);
  return result.hit;
}

}  // namespace

int main() {
  // Endpoints are outside the obstacle in both cases.  Only the crossing
  // transition should report a Bullet continuous hit.
  const bool crossing = sweep_crosses_static_box(true);
  const bool clear = sweep_crosses_static_box(false);
  const bool pass = crossing && !clear;
  std::cout << "{\"fixture\":\"synthetic_static_world_convex_sweep\","
            << "\"endpoint_a_collision\":false,\"endpoint_b_collision\":false,"
            << "\"bullet_sweep_crossing_collision\":" << (crossing ? "true" : "false") << ","
            << "\"bullet_sweep_clear_transition_collision\":" << (clear ? "true" : "false") << ","
            << "\"strict_self_ccd\":\"NOT_AVAILABLE\","
            << "\"status\":\"" << (pass ? "PASS" : "FAIL") << "\"}\n";
  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
