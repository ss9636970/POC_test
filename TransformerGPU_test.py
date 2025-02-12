import torch
from transformers import AutoModel

# 檢查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 載入預訓練模型（這裡使用 BERT）
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)

# 將模型移動到 GPU
model.to(device)

# 顯示 GPU 使用狀況
if device.type == "cuda":
    allocated = torch.cuda.memory_allocated(device) / 1024**2  # 已分配記憶體 (MB)
    reserved = torch.cuda.memory_reserved(device) / 1024**2    # 預留記憶體 (MB)
    print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
else:
    print("No GPU detected.")