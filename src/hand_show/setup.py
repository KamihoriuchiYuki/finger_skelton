from setuptools import find_packages, setup

package_name = 'hand_show'

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
    maintainer='hlab11',
    maintainer_email='ryuozaki21@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "talker_index   = hand_show.talker_index:main",
            "listener_index_raw = hand_show.listener_index_raw:main",
            "listener_index_predicted = hand_show.listener_index_predicted:main",
            "csv_pub        = hand_show.csv_pub:main"
        ],
    },
)
