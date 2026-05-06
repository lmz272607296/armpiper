import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'leapsim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
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
         glob(os.path.join(package_name, 'cfg/train', '*')))

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
        'rot=leapsim.deploy:main',
        'game=leapsim.game:main',
        'hello=leapsim.hello:main',
        'zero=leapsim.ping:main',
        'inference=leapsim.grasp_inference:main',
        ],
    },
)
