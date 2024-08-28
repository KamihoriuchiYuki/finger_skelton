import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/hosodalab9/Sensor-Glove/install/sensor_fusion'
