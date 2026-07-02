#!/usr/bin/env bash
set -eo pipefail

sudo apt-get install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

cat <<'EOF' | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main
EOF

sudo apt-get update
sudo apt-get install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  ros-dev-tools \
  ros-jazzy-desktop \
  ros-jazzy-moveit \
  ros-jazzy-moveit-py

if ! rosdep db 2>/dev/null | grep -q base.yaml; then
  sudo rosdep init || true
fi
rosdep update

grep -q "source /opt/ros/jazzy/setup.bash" ~/.bashrc || {
  echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
}

source /opt/ros/jazzy/setup.bash
ros2 --help >/dev/null
python3 - <<'PY'
from moveit.core.robot_trajectory import RobotTrajectory
print("MoveItPy RobotTrajectory has Ruckig:", hasattr(RobotTrajectory, "apply_ruckig_smoothing"))
PY
