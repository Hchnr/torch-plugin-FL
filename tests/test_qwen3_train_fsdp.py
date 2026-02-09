"""
Qwen3 FSDP (Fully Sharded Data Parallel) Training Test - Using torch_flagos

Usage:
    torchrun --nproc_per_node=2 test_qwen3_train_fsdp.py

    or with more GPUs:
    torchrun --nproc_per_node=4 test_qwen3_train_fsdp.py
"""

import os
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch_flagos  # Automatically registers FlagGems operators to flagos device
import time


def setup_distributed():
    """Initialize distributed training environment"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Set current device
    torch_flagos.flagos.set_device(local_rank)

    return local_rank, world_size, rank


def cleanup_distributed():
    """Clean up distributed training environment"""
    dist.destroy_process_group()


def print_rank0(msg, rank):
    """Only print on rank 0"""
    if rank == 0:
        print(msg)


# Sync function
def sync():
    torch_flagos.flagos.synchronize()


class DummyTextDataset(Dataset):
    """Simple dummy text dataset for testing training workflow"""
    def __init__(self, tokenizer, num_samples=100, max_length=256):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length

        # Generate some simple training samples
        self.texts = [
            "Large language models are neural networks trained on vast amounts of text data.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning uses multiple layers of neural networks.",
            "Transformers revolutionized the field of natural language processing.",
            "Attention mechanisms allow models to focus on relevant parts of the input.",
            "Fine-tuning adapts pre-trained models to specific tasks.",
            "Gradient descent is used to optimize neural network parameters.",
            "Backpropagation computes gradients for training neural networks.",
            "The loss function measures how well the model performs.",
        ] * (num_samples // 10 + 1)
        self.texts = self.texts[:num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.texts[idx % len(self.texts)]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0)  # Autoregressive LM training
        }


def get_fsdp_wrap_policy(model):
    """Get FSDP auto wrap policy for transformer models"""
    # Try to import Qwen3 specific layer class
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        wrap_class = Qwen3DecoderLayer
    except ImportError:
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
            wrap_class = Qwen2DecoderLayer
        except ImportError:
            # Fallback to size-based wrapping
            return functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=1e6  # Wrap modules with at least 1M parameters
            )

    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={wrap_class}
    )


def main():
    # Setup distributed environment
    local_rank, world_size, rank = setup_distributed()
    device = f"flagos:{local_rank}"

    print_rank0("=" * 60, rank)
    print_rank0("torch_flagos Qwen3 FSDP Training Test", rank)
    print_rank0("=" * 60, rank)

    # Check device status
    print_rank0(f"\nFlagos device available: {torch_flagos.flagos.is_available()}", rank)
    print_rank0(f"Device count: {torch_flagos.flagos.device_count()}", rank)
    print_rank0(f"FlagGems registered: {torch_flagos.is_flaggems_enabled()}", rank)
    print_rank0(f"Registered ops count: {len(torch_flagos.get_registered_ops())}", rank)
    print_rank0(f"World size: {world_size}", rank)
    print_rank0(f"Current rank: {rank}, local_rank: {local_rank}", rank)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Configuration parameters
    model_name = "/nfs/hcr/models/Qwen/Qwen3-0.6B"
    BATCH_SIZE = 2
    MAX_SEQ_LEN = 1024
    NUM_TRAIN_STEPS = 10
    LEARNING_RATE = 1e-5
    GRADIENT_ACCUMULATION_STEPS = 1

    # FSDP Configuration
    SHARDING_STRATEGY = ShardingStrategy.FULL_SHARD  # Options: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    USE_CPU_OFFLOAD = False

    # Load model
    print_rank0("\n[1] Loading model and tokenizer...", rank)
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model to CPU first (FSDP will handle sharding)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for training to support gradient computation
        device_map="cpu"
    )

    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    # Synchronize before FSDP initialization
    sync()
    dist.barrier()

    # Get FSDP wrap policy
    auto_wrap_policy = get_fsdp_wrap_policy(model)

    # Configure mixed precision (optional, set to None for full float32)
    mixed_precision_policy = None
    # Uncomment below for mixed precision training:
    # mixed_precision_policy = MixedPrecision(
    #     param_dtype=torch.float32,
    #     reduce_dtype=torch.float32,
    #     buffer_dtype=torch.float32,
    # )

    # Configure CPU offload (optional)
    cpu_offload = CPUOffload(offload_params=True) if USE_CPU_OFFLOAD else None

    # Wrap model with FSDP
    # Note: For custom devices like flagos, use torch.device instead of integer device_id
    model = FSDP(
        model,
        sharding_strategy=SHARDING_STRATEGY,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        cpu_offload=cpu_offload,
        device_id=torch.device(device),
        use_orig_params=True,  # Allows access to original parameter names
    )

    model.train()  # Set to training mode

    print_rank0(f"FSDP Sharding Strategy: {SHARDING_STRATEGY}", rank)
    print_rank0(f"CPU Offload: {USE_CPU_OFFLOAD}", rank)
    print_rank0(f"Model load time: {time.time() - load_start:.2f}s", rank)

    # Print FSDP sharded parameter info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"Total parameters (sharded): {total_params / 1e6:.2f}M", rank)
    print_rank0(f"Trainable parameters (sharded): {trainable_params / 1e6:.2f}M", rank)

    # Create dataset and dataloader with DistributedSampler
    print_rank0("\n[2] Creating dataset...", rank)
    dataset = DummyTextDataset(tokenizer, num_samples=100, max_length=MAX_SEQ_LEN)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        drop_last=True
    )
    print_rank0(f"Dataset size: {len(dataset)}", rank)
    print_rank0(f"Batch size per GPU: {BATCH_SIZE}", rank)
    print_rank0(f"Global batch size: {BATCH_SIZE * world_size}", rank)
    print_rank0(f"Sequence length: {MAX_SEQ_LEN}", rank)

    # Create optimizer
    print_rank0("\n[3] Creating optimizer...", rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print_rank0(f"Optimizer: AdamW, learning rate: {LEARNING_RATE}", rank)

    # Training loop
    print_rank0(f"\n[4] Starting FSDP training ({NUM_TRAIN_STEPS} steps)...", rank)
    print_rank0("    (First run will trigger Triton kernel compilation, may take longer)", rank)

    total_tokens = 0
    total_loss = 0.0
    step_times = []

    sampler.set_epoch(0)
    data_iter = iter(dataloader)

    for step in range(NUM_TRAIN_STEPS):
        # Get data batch
        try:
            batch = next(data_iter)
        except StopIteration:
            sampler.set_epoch(step + 1)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        sync()
        step_start = time.time()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # Debug info (only print on first step and rank 0)
        if step == 0 and rank == 0:
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

            if step == 0 and rank == 0:
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

        print_rank0(f"  Step {step + 1}/{NUM_TRAIN_STEPS}: "
              f"loss={loss.item():.4f}, "
              f"time={step_time:.2f}s, "
              f"tokens/s={tokens_per_sec:.1f}", rank)

    # Summary statistics
    print_rank0("\n" + "=" * 60, rank)
    print_rank0("FSDP Training Summary (torch_flagos + FlagGems):", rank)
    print_rank0(f"  World size: {world_size} GPUs", rank)
    print_rank0(f"  Sharding Strategy: {SHARDING_STRATEGY}", rank)
    print_rank0(f"  Total training steps: {NUM_TRAIN_STEPS}", rank)
    print_rank0(f"  Average loss: {total_loss / NUM_TRAIN_STEPS:.4f}", rank)
    print_rank0(f"  Total tokens (per GPU): {total_tokens}", rank)
    print_rank0(f"  Total tokens (all GPUs): {total_tokens * world_size}", rank)
    print_rank0("-" * 60, rank)

    # Exclude first step (includes compilation overhead)
    if len(step_times) > 1:
        first_step_time = step_times[0]
        rest_step_times = step_times[1:]
        avg_step_time = sum(rest_step_times) / len(rest_step_times)
        tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN

        print_rank0(f"  First step (with compilation): {first_step_time:.2f}s ({tokens_per_step/first_step_time:.1f} tokens/s per GPU)", rank)
        print_rank0(f"  Average subsequent steps: {avg_step_time:.2f}s ({tokens_per_step/avg_step_time:.1f} tokens/s per GPU)", rank)
        print_rank0(f"  Estimated Triton compilation overhead: {first_step_time - avg_step_time:.2f}s", rank)
    else:
        avg_step_time = step_times[0]
        tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
        print_rank0(f"  Average per step: {avg_step_time:.2f}s ({tokens_per_step/avg_step_time:.1f} tokens/s per GPU)", rank)

    print_rank0("-" * 60, rank)
    total_time = sum(step_times)
    print_rank0(f"  Total training time: {total_time:.2f}s", rank)
    print_rank0(f"  Overall throughput (per GPU): {total_tokens / total_time:.1f} tokens/s", rank)
    print_rank0(f"  Overall throughput (all GPUs): {total_tokens * world_size / total_time:.1f} tokens/s", rank)
    print_rank0("=" * 60, rank)

    print_rank0("\nFSDP Training test completed!", rank)

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
