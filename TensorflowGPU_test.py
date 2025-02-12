import tensorflow as tf

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# 顯示 TensorFlow 使用的設備
print("Is GPU Available:", tf.test.is_gpu_available())

# 顯示 GPU 名稱（如果可用）
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print("GPU Device Name:", tf.config.experimental.get_device_details(gpu_devices[0])["device_name"])