# Sensor-Glove
cd ~/hand_sense_ws
colcon build --symlink-install
git clone https://github.com/IntelRealSense/realsense-ros.git src/realsense-ros
source /opt/ros/humble/setup.bash
colcon build
source ~/Sensor-Glove/Sensor-Glove/hand_sense_ws/install/setup.bash

# Install ROS Humble
sudo apt update && sudo apt install curl gnupg lsb-release
sudo apt install curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source .bashrc
sudo apt install python3-colcon-common-extensions

# Install Intel RealSense SDK
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update
sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg

# Install realsense-ros
cd Sensor-Glove/hand_sense_ws/src/
git clone https://github.com/IntelRealSense/realsense-ros.git
cd ~/hand_sense_ws
colcon build --symlink-install
cd src/realsense-ros/
git branch
* ros2-development
#debug
sudo apt-get update
sudo apt-get install ros-humble-diagnostic-updater

# Install pip3
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
pip3 install -U pip

# Install modules
pip3 install pyrealsense2 mediapipe opencv-python opencv-contrib-python 

# Subscribe datas from RealSense
git clone https://github.com/IntelRealSense/realsense-ros.git src/realsense-ros
source /opt/ros/humble/setup.bash
colcon build
source ~/Sensor-Glove/hand_sense_ws/src/install/setup.bash
sudo apt install ros-humble-librealsense2*
sudo apt install ros-humble-realsense2-*
## open another terminal
ros2 topic list

# check
cd ~/Sensor-Glove/hand_sense_ws/src/subscriber/subscriber/


downgrade numpy if numpy >= 2

    pip3 show numpy #check version
    pip3 uninstall numpy
    pip3 install "numpy<2"

# Run with bash
Add below to .bashrc

   source ~/Sensor-Glove/ros2_dev.sh

# Utilize gpu for tensorflow
To use gpu for tensorflow we need cuda, nvidia and cudnn.
See [website](https://www.tensorflow.org/install/pip) for the version of cuda, cudnn and minimum requirement of Nvidia to install.
# Install Nvidia and Cuda
https://qiita.com/porizou1/items/74d8264d6381ee2941bd

## find Nvidia version to install
https://qiita.com/y-vectorfield/items/72bfb66d8ec85847fe2f

check tensorflow version and see CUDA and Nvidia version to install
pip show tensorflow
https://www.tensorflow.org/install/source?hl=ja#gpu
https://www.tensorflow.org/install/pip  

For example, install CUDA 12.3 for https://developer.nvidia.com/cuda-12-3-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local


# Find Arduino

    hlab6@HLAB-6:~$ ls -la /dev/ttyACM0 
    crw-rw---- 1 root dialout 166, 0  8月  8 09:48 /dev/ttyACM0
    hlab6@HLAB-6:~$ sudo chmod 666 /dev/ttyACM0 
    [sudo] password for hlab6: 
    hlab6@HLAB-6:~$ ls -la /dev/ttyACM0 
    crw-rw-rw- 1 root dialout 166, 0  8月  8 09:48 /dev/ttyACM0
