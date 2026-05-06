from setuptools import find_packages, setup


package_name = "vla"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lmz",
    maintainer_email="lmz@todo.todo",
    description="Simple dual-topic test controller for Piper arm and dexterous hand.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "test = vla.test:main",
        ],
    },
)
