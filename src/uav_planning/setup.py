from setuptools import setup, find_packages
import os
from glob import glob

package_name = "uav_planning"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="uav-search-project",
    maintainer_email="todo@todo.com",
    description="Mission planning and spiral search for UAV search simulation",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mission_controller = uav_planning.mission_controller:main",
            "mavsdk_smoke_test = uav_planning.mavsdk_smoke_test:main",
            "sitl_evaluator = uav_planning.sitl_evaluator:main",
        ],
    },
)
