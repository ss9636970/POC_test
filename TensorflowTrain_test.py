import tensorflow as tf
import numpy as np

# 檢查 GPU 是否可用
device_name = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print(f"Using device: {device_name}")

# 生成隨機圖像數據 (1000 張 32x32x3 的圖片，共 10 類)
num_classes = 10
X_train = np.random.rand(1000, 32, 32, 3).astype(np.float32)  # 訓練數據
y_train = np.random.randint(0, num_classes, 1000)  # 訓練標籤

X_test = np.random.rand(200, 32, 32, 3).astype(np.float32)  # 測試數據
y_test = np.random.randint(0, num_classes, 200)  # 測試標籤

# 將標籤轉換為 One-Hot
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# 建立簡單的 CNN 模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練 5 個 epoch
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# 測試模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.2%}")