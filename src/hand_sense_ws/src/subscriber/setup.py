from setuptools import find_packages, setup

package_name = 'subscriber'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hosodalab9',
    maintainer_email='hosodalab9@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "subscriber = subscriber.subscriber:main",
            "angle_finger = subscriber.angle_finger:main"

        ],
    },
)