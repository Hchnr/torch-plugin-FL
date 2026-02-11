"""
Qwen3 Single-GPU Training Test - Using torch_flagos
"""

import torch
import torch_flagos  # Automatically registers FlagGems operators to flagos device
import time

print("=" * 60)
print("torch_flagos Qwen3 Single-GPU Training Test")
print("=" * 60)

# Check device status
print(f"\nFlagos device available: {torch_flagos.flagos.is_available()}")
print(f"Device count: {torch_flagos.flagos.device_count()}")
print(f"FlagGems registered: {torch_flagos.is_flaggems_enabled()}")
print(f"Registered ops count: {len(torch_flagos.get_registered_ops())}")

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from dummy_dataset import DummyTextDataset

# Configuration parameters
model_name = "/nfs/hcr/models/Qwen/Qwen3-0.6B"
DEVICE = "flagos:0"
BATCH_SIZE = 2
MAX_SEQ_LEN = 1024
NUM_TRAIN_STEPS = 10
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 1

# Sync function
def sync():
    torch_flagos.flagos.synchronize()


def main():
    # Load model
    print("\n[1] Loading model and tokenizer...")
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model to CPU, then move to flagos device
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
    print("    (First run will trigger Triton kernel compilation, may take longer)")

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
    print("Training Summary (torch_flagos + FlagGems):")
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

        print(f"  First step (with compilation): {first_step_time:.2f}s ({tokens_per_step/first_step_time:.1f} tokens/s)")
        print(f"  Average subsequent steps: {avg_step_time:.2f}s ({tokens_per_step/avg_step_time:.1f} tokens/s)")
        print(f"  Estimated Triton compilation overhead: {first_step_time - avg_step_time:.2f}s")
    else:
        avg_step_time = step_times[0]
        tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
        print(f"  Average per step: {avg_step_time:.2f}s ({tokens_per_step/avg_step_time:.1f} tokens/s)")

    print("-" * 60)
    total_time = sum(step_times)
    print(f"  Total training time: {total_time:.2f}s")
    print(f"  Overall throughput: {total_tokens / total_time:.1f} tokens/s")
    print("=" * 60)

    print("\nTraining test completed!")


if __name__ == "__main__":
    main()
