from moveit_configs_utils import MoveItConfigsBuilder

cfg = (
    MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
    .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
    .to_moveit_configs()
    .to_dict()
)
print([key for key in cfg if "planning" in key or "ompl" in key])
print("planning_pipelines =", cfg.get("planning_pipelines"))
print("ompl =", cfg.get("ompl"))
