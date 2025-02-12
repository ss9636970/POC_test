import torch
print(torch.cuda.is_available())

### 測試可使用 GPU 數量
print(torch.cuda.device_count())

if torch.cuda.is_available():
  x = torch.rand(3, 3).cuda(0)  # 創建一個隨機 Tensor 並放到 GPU
  y = torch.rand(3, 3).cuda(0)
  print(x + y)