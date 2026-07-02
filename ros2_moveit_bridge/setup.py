from glob import glob
from setuptools import setup


package_name = "fr5_tunnel_moveit_bridge"

setup(
    name=package_name,
    version="0.1.0",
    py_modules=["plan_closed_contour_moveit", "smoke_test_moveit_bridge", "validate_bridge_inputs"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*")),
        (f"share/{package_name}/launch", glob("launch/*launch.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="FR5 tunnel simulation maintainer",
    maintainer_email="maintainer@example.com",
    description="MoveIt2 execution bridge for FR5 closed horseshoe contour tracking.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "plan_closed_contour_moveit = plan_closed_contour_moveit:main",
            "smoke_test_moveit_bridge = smoke_test_moveit_bridge:main",
            "validate_bridge_inputs = validate_bridge_inputs:main",
        ],
    },
)
