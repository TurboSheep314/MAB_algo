from setuptools import setup

package_name = "mab_ucb_bandit"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="ROS 2 package for a 4-armed bandit solved with UCB1.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ucb_bandit_node = mab_ucb_bandit.node:main",
        ],
    },
)
