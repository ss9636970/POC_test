import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 選擇模型（7B 版本）
model_id = "codellama/CodeLlama-7b-hf"

# 檢查 GPU 是否可用，並設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 載入 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda:0"
)

# 建立生成 pipeline
code_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 測試程式碼補全
prompt = "def fibonacci(n):"
output = code_generator(prompt, max_length=100, do_sample=True)

print("Generated Code:")
print(output[0]['generated_text'])