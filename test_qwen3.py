"""
Qwen3 推理测试 - 使用 torch_flagos
"""

import torch
import torch_flagos  # 自动注册 FlagGems 算子到 flagos 设备
import time

print("=" * 60)
print("torch_flagos Qwen3 推理测试")
print("=" * 60)

# 检查设备状态
print(f"\nFlagos 设备可用: {torch_flagos.flagos.is_available()}")
print(f"设备数量: {torch_flagos.flagos.device_count()}")
print(f"FlagGems 已注册: {torch_flagos.is_flaggems_enabled()}")
print(f"已注册算子数: {len(torch_flagos.get_registered_ops())}")

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/nfs/hcr/models/Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 128  # 测试用较小的 token 数
DEVICE = "flagos:0"

# 加载模型
print("\n[1] 加载模型...")
load_start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 先加载到 CPU，再移动到 flagos 设备
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu"  # 先加载到 CPU
)
# 移动模型到 flagos 设备
model = model.to(DEVICE)
print(f"模型设备: {next(model.parameters()).device}")
print(f"模型加载耗时: {time.time() - load_start:.2f}s")

# 准备输入
prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # 关闭 thinking 加快测试
)
model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
input_len = model_inputs.input_ids.shape[1]
print(f"输入 token 数: {input_len}")

# 同步函数
def sync():
    torch_flagos.flagos.synchronize()

# 第一次推理 (包含 Triton 编译)
print(f"\n[2] 第一次推理 - 包含 Triton 算子编译 (max_new_tokens={MAX_NEW_TOKENS})...")
print("    (首次运行会触发 Triton kernel 编译，可能需要较长时间)")
sync()
start1 = time.time()
with torch.no_grad():
    output1 = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS)
sync()
time1 = time.time() - start1
tokens1 = output1.shape[1] - input_len
print(f"第一次推理耗时: {time1:.2f}s, 生成 {tokens1} tokens, 速度: {tokens1/time1:.2f} tokens/s")

# 第二次推理 (使用编译缓存)
print(f"\n[3] 第二次推理 - 使用 Triton 缓存 (max_new_tokens={MAX_NEW_TOKENS})...")
sync()
start2 = time.time()
with torch.no_grad():
    output2 = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS)
sync()
time2 = time.time() - start2
tokens2 = output2.shape[1] - input_len
print(f"第二次推理耗时: {time2:.2f}s, 生成 {tokens2} tokens, 速度: {tokens2/time2:.2f} tokens/s")

# 第三次推理
print(f"\n[4] 第三次推理 (max_new_tokens={MAX_NEW_TOKENS})...")
sync()
start3 = time.time()
with torch.no_grad():
    output3 = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS)
sync()
time3 = time.time() - start3
tokens3 = output3.shape[1] - input_len
print(f"第三次推理耗时: {time3:.2f}s, 生成 {tokens3} tokens, 速度: {tokens3/time3:.2f} tokens/s")

# 第四次推理
print(f"\n[5] 第四次推理 (max_new_tokens={MAX_NEW_TOKENS})...")
sync()
start4 = time.time()
with torch.no_grad():
    output4 = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS)
sync()
time4 = time.time() - start4
tokens4 = output4.shape[1] - input_len
print(f"第四次推理耗时: {time4:.2f}s, 生成 {tokens4} tokens, 速度: {tokens4/time4:.2f} tokens/s")

# 汇总
print("\n" + "=" * 60)
print("汇总 (torch_flagos + FlagGems):")
print(f"  第一次 (含编译): {time1:.2f}s ({tokens1/time1:.2f} tokens/s)")
print(f"  第二次 (使用缓存): {time2:.2f}s ({tokens2/time2:.2f} tokens/s)")
print(f"  第三次: {time3:.2f}s ({tokens3/time3:.2f} tokens/s)")
print(f"  第四次: {time4:.2f}s ({tokens4/time4:.2f} tokens/s)")
print(f"  平均 (2-4次): {(time2+time3+time4)/3:.2f}s ({(tokens2+tokens3+tokens4)/(time2+time3+time4):.2f} tokens/s)")
print("-" * 60)
print(f"  Triton 编译开销估算: {time1 - (time2+time3+time4)/3:.2f}s")
print("=" * 60)

# 输出生成内容
print("\n生成内容:")
print(tokenizer.decode(output4[0][input_len:], skip_special_tokens=True))
