"""
Qwen3 Inference Test - Using torch_flagos
"""

import torch
import torch_flagos  # Automatically registers FlagGems operators to flagos device
import time

print("=" * 60)
print("torch_flagos Qwen3 Inference Test")
print("=" * 60)

# Check device status
print(f"\nFlagos device available: {torch_flagos.flagos.is_available()}")
print(f"Device count: {torch_flagos.flagos.device_count()}")
print(f"FlagGems registered: {torch_flagos.is_flaggems_enabled()}")
print(f"Registered ops count: {len(torch_flagos.get_registered_ops())}")

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/nfs/hcr/models/Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 128  # Smaller token count for testing
DEVICE = "flagos:0"

# Load model
print("\n[1] Loading model...")
load_start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load to CPU first, then move to flagos device
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu"  # Load to CPU first
)
# Move model to flagos device
model = model.to(DEVICE)
print(f"Model device: {next(model.parameters()).device}")
print(f"Model load time: {time.time() - load_start:.2f}s")

# Prepare input
prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # Disable thinking to speed up test
)
model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
input_len = model_inputs.input_ids.shape[1]
print(f"Input token count: {input_len}")

# Sync function
def sync():
    torch_flagos.flagos.synchronize()

# First inference (includes Triton compilation)
print(f"\n[2] First inference - includes Triton kernel compilation (max_new_tokens={MAX_NEW_TOKENS})...")
print("    (First run will trigger Triton kernel compilation, may take longer)")
sync()
start1 = time.time()
with torch.no_grad():
    output1 = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS)
sync()
time1 = time.time() - start1
tokens1 = output1.shape[1] - input_len
print(f"First inference time: {time1:.2f}s, generated {tokens1} tokens, speed: {tokens1/time1:.2f} tokens/s")

# Second inference (using compilation cache)
print(f"\n[3] Second inference - using Triton cache (max_new_tokens={MAX_NEW_TOKENS})...")
sync()
start2 = time.time()
with torch.no_grad():
    output2 = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS)
sync()
time2 = time.time() - start2
tokens2 = output2.shape[1] - input_len
print(f"Second inference time: {time2:.2f}s, generated {tokens2} tokens, speed: {tokens2/time2:.2f} tokens/s")

# Third inference
print(f"\n[4] Third inference (max_new_tokens={MAX_NEW_TOKENS})...")
sync()
start3 = time.time()
with torch.no_grad():
    output3 = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS)
sync()
time3 = time.time() - start3
tokens3 = output3.shape[1] - input_len
print(f"Third inference time: {time3:.2f}s, generated {tokens3} tokens, speed: {tokens3/time3:.2f} tokens/s")

# Fourth inference
print(f"\n[5] Fourth inference (max_new_tokens={MAX_NEW_TOKENS})...")
sync()
start4 = time.time()
with torch.no_grad():
    output4 = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS)
sync()
time4 = time.time() - start4
tokens4 = output4.shape[1] - input_len
print(f"Fourth inference time: {time4:.2f}s, generated {tokens4} tokens, speed: {tokens4/time4:.2f} tokens/s")

# Summary
print("\n" + "=" * 60)
print("Summary (torch_flagos + FlagGems):")
print(f"  First (with compilation): {time1:.2f}s ({tokens1/time1:.2f} tokens/s)")
print(f"  Second (using cache): {time2:.2f}s ({tokens2/time2:.2f} tokens/s)")
print(f"  Third: {time3:.2f}s ({tokens3/time3:.2f} tokens/s)")
print(f"  Fourth: {time4:.2f}s ({tokens4/time4:.2f} tokens/s)")
print(f"  Average (runs 2-4): {(time2+time3+time4)/3:.2f}s ({(tokens2+tokens3+tokens4)/(time2+time3+time4):.2f} tokens/s)")
print("-" * 60)
print(f"  Estimated Triton compilation overhead: {time1 - (time2+time3+time4)/3:.2f}s")
print("=" * 60)

# Output generated content
print("\nGenerated content:")
print(tokenizer.decode(output4[0][input_len:], skip_special_tokens=True))
