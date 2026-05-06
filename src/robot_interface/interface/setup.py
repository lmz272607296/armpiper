import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    package_data={
        package_name: [
            'cfg/*.yaml',
            'cfg/task/*',
            'cfg/train/*',
            'cache/*',
            'runs/*',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'cache'), 
         glob(os.path.join(package_name, 'cache', '*'))),

        (os.path.join('share', package_name, 'cfg'), glob(os.path.join(package_name, 'cfg', '*.yaml'))),

        (os.path.join('share', package_name, 'cfg/task'), 
         glob(os.path.join(package_name,'cfg/task', '*'))),


        (os.path.join('share', package_name, 'cfg/train'), 
         glob(os.path.join(package_name, 'cfg/train', '*'))),

        (os.path.join('share', package_name, 'assets/urdf'),
         glob(os.path.join('..', 'assets', 'urdf', '*.urdf')))
        ,
        (os.path.join('share', package_name, 'assets/meshes'),
         glob(os.path.join('..', 'assets', 'meshes', '*')))

    ],
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lmz',
    maintainer_email='lmz@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'ultimate_yolo_node_cpu=interface.ultimate_yolo_node_cpu:main',
        'action1_4_move=interface.action1_4_move:main',
        'action5_rot=interface.action5_rot:main',
        'action6_gsp=interface.action6_gsp:main',
        'action7_flat=interface.action7_flat:main',
        'action8_rst=interface.action8_rst:main',
        'master_node=interface.master_node:main',
        ],
    },
)
