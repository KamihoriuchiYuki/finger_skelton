import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# run python3 check_gpu.py and if the output is device_type: "GPU" then the GPU is available