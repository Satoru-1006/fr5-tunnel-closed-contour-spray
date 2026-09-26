# D41 method ledger

- Canonical algorithm retained: D39 stable residual prior followed by D40 native MoveIt FK/Jacobian position-dominant DLS (`routeA_DLS_position_dominant_v4`).
- Native collision layer: C++ MoveIt2 `CollisionEnvBullet::checkRobotCollision(req,result,state1,state2,acm)` for robot-world segments; this is a real two-state Bullet continuous query, not a renamed dense scan.
- Self collision: native PlanningScene self queries plus adaptive 0.5-degree joint-space refinement; the installed MoveIt Bullet wrapper does not expose a self two-state continuous overload.
- Clearance: native `distanceRobot` and `distanceSelf`, with pair names and state/waypoint indices in `D41_CLEARANCE_REPORT.csv`; sampled/refined distances are labelled as such.
- Numerical health: native RobotState Jacobian SVD and optional D40 DLS per-iteration instrumentation.
- Robustness: 37 deterministic shadow cases across initial state, Cartesian translation, normal, density, curvature, near-singularity, and near-limit families.
- Native API references: MoveIt2 PlanningScene collision/distance interfaces ([planning_scene.hpp](https://github.com/moveit/moveit2/blob/main/moveit_core/planning_scene/include/moveit/planning_scene/planning_scene.hpp)); MoveIt FCL distance implementation ([collision_env_fcl.cpp](https://github.com/moveit/moveit/blob/master/moveit_core/collision_detection_fcl/src/collision_env_fcl.cpp)); FCL continuous-collision primitives ([FCL](https://github.com/flexible-collision-library/fcl)).
- Promotion: the native CCD/clearance layer and the explicit URDF-bound joint-limit-aware DLS projection are promoted only when nominal, adversarial, numerical-health, and robustness gates all pass; the default D40 route remains unchanged outside D41.
