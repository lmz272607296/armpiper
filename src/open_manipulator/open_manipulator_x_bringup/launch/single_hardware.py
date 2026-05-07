import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import LaunchConfiguration
from launch.substitutions import ThisLaunchFileDir

from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    start_rviz = LaunchConfiguration('start_rviz')
    prefix = LaunchConfiguration('prefix')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    use_sim = LaunchConfiguration('use_sim')

    # ################################################################### #
    # #                  MODIFICATION START                             # #
    # ################################################################### #
    # Auto-detect serial port
    port_to_use = '/dev/ttyUSB0'
    if not os.path.exists(port_to_use):
        print(f"INFO: Port '{port_to_use}' not found, attempting to use '/dev/ttyUSB1'.")
        port_to_use = '/dev/ttyUSB1'
        if not os.path.exists(port_to_use):
            print(f"ERROR: Neither '/dev/ttyUSB0' nor '/dev/ttyUSB1' found.")
            print(f"ERROR: Defaulting to '/dev/ttyUSB0'. The hardware node will likely fail.")
            # Fallback to default
            port_to_use = '/dev/ttyUSB0'

    hand_port_name = LaunchConfiguration('hand_port_name', default=port_to_use)
    # ################################################################### #
    # #                  MODIFICATION END                               # #
    # ################################################################### #

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_rviz',
            default_value='false',
            description='Whether execute rviz2'),

        DeclareLaunchArgument(
            'prefix',
            default_value='""',
            description='Prefix of the joint and link names'),

        DeclareLaunchArgument(
            'use_fake_hardware',
            default_value='false',
            description='Start robot with fake hardware mirroring command to its states.'),

        DeclareLaunchArgument(
            'use_sim',
            default_value='false',
            description='Start robot in Gazebo simulation.'),

        DeclareLaunchArgument(
            'hand_port_name',
            default_value=port_to_use,
            description='The port name for the hand hardware. Auto-detected from /dev/ttyUSB0 or /dev/ttyUSB1.'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([ThisLaunchFileDir(), '/base.launch.py']),
            launch_arguments={
                'start_rviz': start_rviz,
                'prefix': prefix,
                'use_fake_hardware': use_fake_hardware,
                'use_sim': use_sim,
                'hand_port_name': hand_port_name,
            }.items(),
        ),
    ])
