"""
Qwen3 Single-GPU Training Baseline - Pure CUDA (no flagos/FlagGems)

This script uses native CUDA device directly for performance baseline comparison.
No FlagGems Triton kernels are used; all ops run through PyTorch's default CUDA kernels.

Usage:
    python tests/test_qwen3_train_baseline.py

On A800 with divice=CUDA:
  Step 1/10: loss=14.4110, time=0.32s, tokens/s=6435.4
  Step 2/10: loss=7.4065, time=0.25s, tokens/s=8234.8
  Step 3/10: loss=2.7476, time=0.25s, tokens/s=8221.4
  Step 4/10: loss=0.9689, time=0.25s, tokens/s=8296.8
  Step 5/10: loss=2.0250, time=0.25s, tokens/s=8266.3
  Step 6/10: loss=0.0954, time=0.25s, tokens/s=8221.6
  Step 7/10: loss=0.0402, time=0.25s, tokens/s=8287.8
  Step 8/10: loss=0.0382, time=0.25s, tokens/s=8236.1
  Step 9/10: loss=0.0340, time=0.25s, tokens/s=8268.9
  Step 10/10: loss=0.0453, time=0.25s, tokens/s=8285.4
"""

import torch
import time

print("=" * 60)
print("Qwen3 Single-GPU Training Baseline (Pure CUDA)")
print("=" * 60)

# Check CUDA status
print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from dummy_dataset import DummyTextDataset

# Configuration parameters
model_name = "/nfs/hcr/models/Qwen/Qwen3-0.6B"
DEVICE = "cuda:0"
BATCH_SIZE = 2
MAX_SEQ_LEN = 1024
NUM_TRAIN_STEPS = 10
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 1

# Sync function
def sync():
    torch.cuda.synchronize()


def main():
    # Load model
    print("\n[1] Loading model and tokenizer...")
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model directly to CUDA
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for training to support gradient computation
        device_map="cpu"
    )
    model = model.to(DEVICE)
    model.train()  # Set to training mode

    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    # Pre-warmup: Detect and freeze unused parameters.
    # This is necessary because some parameters (e.g. in rotary embeddings) may not
    # receive gradients during training. Freezing them avoids errors during backward pass.
    print("\n[1.5] Detecting and freezing unused parameters...")

    # Do a forward + backward pass to detect which parameters don't receive gradients
    dummy_input = torch.randint(0, 1000, (1, 32), device=DEVICE)
    with torch.enable_grad():
        dummy_output = model(input_ids=dummy_input, attention_mask=None, labels=None, use_cache=False)
        dummy_loss = dummy_output.logits.sum()
        dummy_loss.backward()

    # Find and freeze parameters that didn't receive gradients
    unused_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            param.requires_grad = False
            unused_params.append(name)
        else:
            param.grad = None  # Clear gradient for actual training

    print(f"    Frozen {len(unused_params)} unused parameters")
    if unused_params:
        for name in unused_params[:5]:
            print(f"      - {name}")
        if len(unused_params) > 5:
            print(f"      ... and {len(unused_params) - 5} more")

    sync()

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"Model load time: {time.time() - load_start:.2f}s")

    # Create dataset and dataloader
    print("\n[2] Creating dataset...")
    dataset = DummyTextDataset(tokenizer, num_samples=100, max_length=MAX_SEQ_LEN)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Sequence length: {MAX_SEQ_LEN}")

    # Create optimizer
    print("\n[3] Creating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"Optimizer: AdamW, learning rate: {LEARNING_RATE}")

    # Training loop
    print(f"\n[4] Starting training ({NUM_TRAIN_STEPS} steps)...")

    total_tokens = 0
    total_loss = 0.0
    step_times = []

    data_iter = iter(dataloader)

    for step in range(NUM_TRAIN_STEPS):
        # Get data batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move data to device
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        sync()
        step_start = time.time()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # Debug info (only print on first step)
        if step == 0:
            print(f"    [DEBUG] outputs.loss: {outputs.loss}")
            print(f"    [DEBUG] outputs.loss.requires_grad: {outputs.loss.requires_grad}")
            print(f"    [DEBUG] outputs.logits.requires_grad: {outputs.logits.requires_grad}")

        # If outputs.loss has no gradient, compute loss manually
        if outputs.loss.requires_grad:
            loss = outputs.loss
        else:
            # Manually compute cross-entropy loss
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            if step == 0:
                print(f"    [DEBUG] Manual loss: {loss}")
                print(f"    [DEBUG] Manual loss.requires_grad: {loss.requires_grad}")

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        sync()
        step_time = time.time() - step_start
        step_times.append(step_time)

        # Statistics
        batch_tokens = input_ids.numel()
        total_tokens += batch_tokens
        total_loss += loss.item()

        tokens_per_sec = batch_tokens / step_time

        print(f"  Step {step + 1}/{NUM_TRAIN_STEPS}: "
              f"loss={loss.item():.4f}, "
              f"time={step_time:.2f}s, "
              f"tokens/s={tokens_per_sec:.1f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Training Summary (Pure CUDA Baseline):")
    print(f"  Total training steps: {NUM_TRAIN_STEPS}")
    print(f"  Average loss: {total_loss / NUM_TRAIN_STEPS:.4f}")
    print(f"  Total tokens: {total_tokens}")
    print("-" * 60)

    # Exclude first step (includes compilation overhead)
    if len(step_times) > 1:
        first_step_time = step_times[0]
        rest_step_times = step_times[1:]
        avg_step_time = sum(rest_step_times) / len(rest_step_times)
        tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN

        print(f"  First step: {first_step_time:.2f}s ({tokens_per_step/first_step_time:.1f} tokens/s)")
        print(f"  Average subsequent steps: {avg_step_time:.2f}s ({tokens_per_step/avg_step_time:.1f} tokens/s)")
    else:
        avg_step_time = step_times[0]
        tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
        print(f"  Average per step: {avg_step_time:.2f}s ({tokens_per_step/avg_step_time:.1f} tokens/s)")

    print("-" * 60)
    total_time = sum(step_times)
    print(f"  Total training time: {total_time:.2f}s")
    print(f"  Overall throughput: {total_tokens / total_time:.1f} tokens/s")
    print("=" * 60)

    print("\nBaseline training test completed!")


if __name__ == "__main__":
    main()
