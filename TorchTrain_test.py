import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda  # 用於獲取 GPU 資訊

# 設置設備（使用 GPU 如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 生成隨機圖像數據 (1000 張 3x32x32 的圖片，共 10 類)
num_classes = 10
X_train = torch.rand(1000, 3, 32, 32).to(device)  # 訓練數據
y_train = torch.randint(0, num_classes, (1000,)).to(device)  # 標籤

X_test = torch.rand(200, 3, 32, 32).to(device)  # 測試數據
y_test = torch.randint(0, num_classes, (200,)).to(device)  # 測試標籤

# 定義簡單的 CNN 模型
class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc = nn.Linear(16 * 16 * 16, num_classes)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv(x)))
    x = x.view(x.size(0), -1)  # 攤平
    x = self.fc(x)
    return x

# 初始化模型
model = SimpleCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 訓練 5 個 epoch
for epoch in range(10):
  optimizer.zero_grad()
  outputs = model(X_train)
  loss = loss_fn(outputs, y_train)
  loss.backward()
  optimizer.step()
  print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

  if device.type == "cuda":
    allocated = cuda.memory_allocated(device) / 1024**2  # 已分配記憶體 (MB)
    reserved = cuda.memory_reserved(device) / 1024**2    # 預留記憶體 (MB)
    print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

# 測試模型
with torch.no_grad():
  test_outputs = model(X_test)
  _, predicted = torch.max(test_outputs, 1)
  accuracy = (predicted == y_test).float().mean().item()
  print(f"Test Accuracy: {accuracy:.2%}")