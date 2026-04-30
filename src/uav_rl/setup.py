from glob import glob
import os

from setuptools import find_packages, setup


package_name = "uav_rl"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="uav-search-project",
    maintainer_email="todo@todo.com",
    description="Gymnasium task environments and learned UAV search policies",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "train_search_policy = uav_rl.train_search_policy:main",
            "train_search_policy_v2 = uav_rl.train_search_policy_v2:main",
            "eval_search_policy = uav_rl.eval_search_policy:main",
            "eval_search_policy_v2 = uav_rl.eval_search_policy_v2:main",
            "rl_mission_controller = uav_rl.rl_mission_controller:main",
        ],
    },
)
