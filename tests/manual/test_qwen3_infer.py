"""
Qwen3 Inference Test - Using torch_flagos (MACA manual test)

Usage:
    python tests/manual/test_qwen3_infer.py
    LD_LIBRARY_PATH=/opt/maca-3.3.0/tools/cu-bridge/lib:$LD_LIBRARY_PATH python tests/manual/test_qwen3_infer.py
"""

import torch_flagos  # Must be imported before torch on MACA (loads cudart shim)
import torch
import time

print("=" * 60)
print("torch_flagos Qwen3 Inference Test")
print("=" * 60)

print(f"\nFlagos device available: {torch_flagos.flagos.is_available()}")
print(f"Device count: {torch_flagos.flagos.device_count()}")
print(f"FlagGems registered: {torch_flagos.is_flaggems_enabled()}")
print(f"Registered ops count: {len(torch_flagos.get_registered_ops())}")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

model_name = "/nfs/hcr/models/Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 128
DEVICE = "flagos:0"

print("\n[1] Loading model...")
load_start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="cpu"
)
model = model.to(DEVICE)
model.eval()
print(f"Model device: {next(model.parameters()).device}")
print(f"Model load time: {time.time() - load_start:.2f}s")

prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
input_len = model_inputs.input_ids.shape[1]
print(f"Input token count: {input_len}")


def sync():
    torch_flagos.flagos.synchronize()


times = []
token_counts = []
run_labels = [
    "First (includes Triton kernel compilation)",
    "Second (using Triton cache)",
    "Third",
    "Fourth",
]
for i in range(4):
    print(f"\n[{i + 2}] {run_labels[i]} (max_new_tokens={MAX_NEW_TOKENS})...")
    if i == 0:
        print("    (First run will trigger Triton kernel compilation, may take longer)")
    sync()
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS)
    sync()
    elapsed = time.time() - t0
    new_tokens = output.shape[1] - input_len
    times.append(elapsed)
    token_counts.append(new_tokens)
    print(
        f"{run_labels[i]}: {elapsed:.2f}s, {new_tokens} tokens, {new_tokens / elapsed:.2f} tokens/s"
    )

print("\n" + "=" * 60)
print("Summary (torch_flagos + FlagGems):")
for label, t, tc in zip(run_labels, times, token_counts):
    print(f"  {label}: {t:.2f}s ({tc / t:.2f} tokens/s)")
avg_t = sum(times[1:]) / 3
avg_tps = sum(token_counts[1:]) / sum(times[1:])
print(f"  Average (runs 2-4): {avg_t:.2f}s ({avg_tps:.2f} tokens/s)")
print(f"  Estimated Triton compilation overhead: {times[0] - avg_t:.2f}s")
print("=" * 60)

print("\nGenerated content:")
print(tokenizer.decode(output[0][input_len:], skip_special_tokens=True))
