"""
CUDA Platform - Qwen3 End-to-End Inference Test

Usage:
    python tests/cuda/test_qwen3_infer.py
    python tests/cuda/test_qwen3_infer.py --model /path/to/Qwen3-0.6B
"""

import argparse
import sys
import time

import torch

print("=" * 60)
print("CUDA Qwen3 Inference Test")
print("=" * 60)

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available.")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/nfs/hcr/models/Qwen/Qwen3-0.6B")
parser.add_argument("--max-new-tokens", type=int, default=128)
args = parser.parse_args()

DEVICE = "cuda:0"
print(f"\nDevice: {torch.cuda.get_device_name(0)}")
print(f"Model:  {args.model}")

from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Load ---
print("\n[1] Loading model...")
load_start = time.time()
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    device_map=DEVICE,
)
model.eval()
print(f"    Device: {next(model.parameters()).device}")
print(f"    Load time: {time.time() - load_start:.2f}s")

# --- Prepare input ---
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
print(f"    Input tokens: {input_len}")


def sync():
    torch.cuda.synchronize()


# --- Inference runs ---
times = []
token_counts = []
for i in range(4):
    label = ["First (may include JIT warm-up)", "Second", "Third", "Fourth"][i]
    print(f"\n[{i+2}] {label} inference (max_new_tokens={args.max_new_tokens})...")
    sync()
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens)
    sync()
    elapsed = time.time() - t0
    new_tokens = output.shape[1] - input_len
    times.append(elapsed)
    token_counts.append(new_tokens)
    print(f"    {elapsed:.2f}s, {new_tokens} tokens, {new_tokens/elapsed:.2f} tok/s")

# --- Summary ---
print("\n" + "=" * 60)
print("Summary:")
labels = ["First", "Second", "Third", "Fourth"]
for i, (t, tc) in enumerate(zip(times, token_counts)):
    print(f"  {labels[i]}: {t:.2f}s ({tc/t:.2f} tok/s)")
avg_t = sum(times[1:]) / 3
avg_tps = sum(token_counts[1:]) / sum(times[1:])
print(f"  Average (runs 2-4): {avg_t:.2f}s ({avg_tps:.2f} tok/s)")
print("=" * 60)

print("\nGenerated output:")
print(tokenizer.decode(output[0][input_len:], skip_special_tokens=True))
