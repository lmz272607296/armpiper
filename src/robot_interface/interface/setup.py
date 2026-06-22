import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'interface'


def glob_files(pattern):
    return [path for path in glob(pattern) if os.path.isfile(path)]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    package_data={
        package_name: [
            'data/*',
            'cfg/*.yaml',
            'cfg/task/*',
            'cfg/train/*',
            'cache/*',
            'runs/*',
            'weights/*',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data'),
         glob_files(os.path.join(package_name, 'data', '*'))),
        (os.path.join('share', package_name, 'cache'), 
         glob_files(os.path.join(package_name, 'cache', '*'))),

        (os.path.join('share', package_name, 'cfg'), glob_files(os.path.join(package_name, 'cfg', '*.yaml'))),

        (os.path.join('share', package_name, 'cfg/task'), 
         glob_files(os.path.join(package_name,'cfg/task', '*'))),


        (os.path.join('share', package_name, 'cfg/train'), 
         glob_files(os.path.join(package_name, 'cfg/train', '*'))),

        (os.path.join('share', package_name, 'assets/urdf'),
         glob_files(os.path.join('..', 'assets', 'urdf', '*.urdf')))
        ,
        (os.path.join('share', package_name, 'assets/meshes'),
         glob_files(os.path.join('..', 'assets', 'meshes', '*')))

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
          'hand_inference_controller=interface.run_inference:main_ros',
          'position_player=interface.position_player:main',
         'position_record=interface.position_record:main',
          'action1_4_move=interface.action1_4_move:main',
          'action5_rot=interface.action5_rot:main',
          'action6_gsp=interface.action6_gsp:main',
        'action7_flat=interface.action7_flat:main',
        'action8_rst=interface.action8_rst:main',
        'eye_in_hand_calibration_node=interface.eye_in_hand_calibration_node:main',
        'chessboard_hand_eye_calibration_node=interface.chessboard_hand_eye_calibration_node:main',
        'camera_base_transform_node=interface.camera_base_transform_node:main',
        'piper_keyboard_joint_jog=interface.piper_keyboard_joint_jog:main',
        'master_node=interface.master_node:main',
        ],
    },
)
