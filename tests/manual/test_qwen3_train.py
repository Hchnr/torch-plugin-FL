"""
Qwen3 Training Test - Unified script supporting multiple configurations.

Supports:
  - Device: cuda (baseline) or flagos (FlagGems Triton kernels)
  - Parallel: none (single GPU), ddp, or fsdp
  - Communication backend: nccl or flagcx (for ddp/fsdp)

Usage:
    # Single GPU, pure CUDA baseline
    python tests/test_qwen3_train.py --device cuda

    # Single GPU, flagos (FlagGems)
    python tests/test_qwen3_train.py --device flagos

    # DDP with NCCL
    torchrun --nproc_per_node=2 tests/test_qwen3_train.py --parallel ddp --comm nccl

    # DDP with FlagCX
    torchrun --nproc_per_node=2 tests/test_qwen3_train.py --parallel ddp --comm flagcx

    # FSDP with NCCL
    torchrun --nproc_per_node=2 tests/test_qwen3_train.py --parallel fsdp --comm nccl

    # FSDP with FlagCX
    torchrun --nproc_per_node=2 tests/test_qwen3_train.py --parallel fsdp --comm flagcx
"""

import argparse
import functools
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from dummy_dataset import DummyTextDataset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 Training Test")
    parser.add_argument(
        "--device",
        choices=["cuda", "flagos"],
        default="flagos",
        help="Device type (default: flagos)",
    )
    parser.add_argument(
        "--parallel",
        choices=["none", "ddp", "fsdp"],
        default="none",
        help="Parallel strategy (default: none)",
    )
    parser.add_argument(
        "--comm",
        choices=["nccl", "flagcx"],
        default="nccl",
        help="Communication backend for distributed (default: nccl)",
    )
    parser.add_argument(
        "--model", default="/nfs/hcr/models/Qwen/Qwen3-0.6B", help="Model path"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length (default: 1024 for single, 256 for distributed)",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    if args.seq_len is None:
        args.seq_len = 256 if args.parallel != "none" else 1024
    return args


# ---------------------------------------------------------------------------
# Sync & printing utilities
# ---------------------------------------------------------------------------


def sync(args):
    if args.device == "flagos":
        import torch_flagos

        torch_flagos.flagos.synchronize()
    else:
        torch.cuda.synchronize()


def print_rank0(msg, rank):
    if rank == 0:
        print(msg)


# ---------------------------------------------------------------------------
# Device & distributed setup
# ---------------------------------------------------------------------------


def setup(args):
    """Initialize device and (optionally) distributed environment.

    Returns (device_str, local_rank, world_size, rank).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    if args.device == "flagos":
        import torch_flagos

        torch_flagos.flagos.set_device(local_rank)
    else:
        torch.cuda.set_device(local_rank)

    if args.parallel == "none":
        return f"{args.device}:{local_rank}", local_rank, 1, 0

    # --- Distributed init ---
    if args.device == "flagos":
        import torch_flagos.distributed as flagos_dist

        flagos_dist.init_process_group(backend=args.comm)
    else:
        if args.comm == "flagcx":
            import flagcx  # noqa: F401

            dist.init_process_group(backend="cpu:gloo,cuda:flagcx")
        else:
            dist.init_process_group(backend="nccl")

    if rank == 0:
        pg = dist.distributed_c10d._get_default_group()
        print(f"[DEBUG] Backend config: {dist.get_backend_config()}", flush=True)
        print(f"[DEBUG] Process group device types: {pg._device_types}", flush=True)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = f"{args.device}:{local_rank}"
    return device, local_rank, world_size, rank


def cleanup(args):
    if args.parallel != "none":
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(args, device, rank):
    """Load model and tokenizer, detect & freeze unused params."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print_rank0("\n[1] Loading model and tokenizer...", rank)
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(torch_dtype=torch.float32, device_map="cpu")
    load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model = model.to(device)
    model.train()

    # Detect and freeze unused parameters
    print_rank0("\n[1.5] Detecting and freezing unused parameters...", rank)
    dummy_input = torch.randint(0, 1000, (1, 32), device=device)
    with torch.enable_grad():
        out = model(
            input_ids=dummy_input, attention_mask=None, labels=None, use_cache=False
        )
        out.logits.sum().backward()

    unused_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            param.requires_grad = False
            unused_params.append(name)
        else:
            param.grad = None

    print_rank0(f"    Frozen {len(unused_params)} unused parameters", rank)
    if rank == 0 and unused_params:
        for name in unused_params[:5]:
            print(f"      - {name}")
        if len(unused_params) > 5:
            print(f"      ... and {len(unused_params) - 5} more")

    sync(args)
    print_rank0(f"Model device: {next(model.parameters()).device}", rank)
    print_rank0(
        f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
        rank,
    )
    print_rank0(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M",
        rank,
    )
    print_rank0(f"Model load time: {time.time() - load_start:.2f}s", rank)

    return model, tokenizer


# ---------------------------------------------------------------------------
# DDP wrapping
# ---------------------------------------------------------------------------


def wrap_ddp(model, args, local_rank, rank):
    """Wrap model with DDP."""
    if args.device == "flagos":
        import torch_flagos.distributed as flagos_dist

        model = flagos_dist.DistributedDataParallel(model)
        print_rank0("    DDP: flagos mode (python_reducer + custom grad hooks)", rank)
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[local_rank])
        print_rank0("    DDP: standard mode (CUDA)", rank)
    return model


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------


def wrap_fsdp(model, args, device, rank):
    """Wrap model with FSDP."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer}
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.device(device),
        use_orig_params=True,
    )
    model.train()

    # Validate: detect unused parameters via dummy forward+backward
    print_rank0("\n[1.5b] Validating FSDP gradient flow...", rank)
    dummy_input = torch.randint(0, 1000, (1, 32), device=device)
    with torch.enable_grad():
        out = model(input_ids=dummy_input, use_cache=False)
        out.logits.sum().backward()

    unused = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    print_rank0(f"    Parameters without gradient: {len(unused)}", rank)
    model.zero_grad(set_to_none=True)

    print_rank0(f"    FSDP: FULL_SHARD (device={args.device})", rank)
    return model


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


def create_dataloader(args, tokenizer, world_size, rank):
    """Create dataloader (with DistributedSampler if distributed)."""
    dataset = DummyTextDataset(tokenizer, num_samples=100, max_length=args.seq_len)
    sampler = None
    if args.parallel != "none":
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )
    return dataloader, sampler


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_step(model, batch, device, args):
    """Forward + loss computation.

    Returns (loss, batch_tokens).
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    loss = outputs.loss

    return loss, input_ids.numel()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(args, step_times, total_loss, total_tokens, world_size, rank):
    label_parts = []
    if args.device == "flagos":
        label_parts.append("flagos + FlagGems")
    else:
        label_parts.append("Pure CUDA")
    if args.parallel != "none":
        label_parts.append(args.parallel.upper())
    if args.parallel != "none":
        label_parts.append(args.comm.upper())
    label = " | ".join(label_parts)

    print_rank0("\n" + "=" * 60, rank)
    print_rank0(f"Training Summary ({label}):", rank)
    if args.parallel != "none":
        print_rank0(f"  World size: {world_size} GPUs", rank)
    print_rank0(f"  Total training steps: {args.steps}", rank)
    print_rank0(f"  Average loss: {total_loss / args.steps:.4f}", rank)
    if args.parallel != "none":
        print_rank0(f"  Total tokens (per GPU): {total_tokens}", rank)
        print_rank0(f"  Total tokens (all GPUs): {total_tokens * world_size}", rank)
    else:
        print_rank0(f"  Total tokens: {total_tokens}", rank)
    print_rank0("-" * 60, rank)

    tokens_per_step = args.batch_size * args.seq_len
    suffix = " per GPU" if args.parallel != "none" else ""

    if len(step_times) > 1:
        first = step_times[0]
        rest = step_times[1:]
        avg = sum(rest) / len(rest)
        print_rank0(
            f"  First step: {first:.2f}s ({tokens_per_step / first:.1f} tokens/s{suffix})",
            rank,
        )
        print_rank0(
            f"  Average subsequent steps: {avg:.2f}s ({tokens_per_step / avg:.1f} tokens/s{suffix})",
            rank,
        )
    else:
        avg = step_times[0]
        print_rank0(
            f"  Average per step: {avg:.2f}s ({tokens_per_step / avg:.1f} tokens/s{suffix})",
            rank,
        )

    print_rank0("-" * 60, rank)
    total_time = sum(step_times)
    print_rank0(f"  Total training time: {total_time:.2f}s", rank)
    print_rank0(
        f"  Overall throughput{suffix}: {total_tokens / total_time:.1f} tokens/s", rank
    )
    if args.parallel != "none":
        print_rank0(
            f"  Overall throughput (all GPUs): {total_tokens * world_size / total_time:.1f} tokens/s",
            rank,
        )
    print_rank0("=" * 60, rank)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # --- Setup ---
    device, local_rank, world_size, rank = setup(args)

    label_parts = [args.device.upper()]
    if args.parallel != "none":
        label_parts.append(args.parallel.upper())
        label_parts.append(args.comm.upper())

    print_rank0("=" * 60, rank)
    print_rank0(f"Qwen3 Training Test [{' | '.join(label_parts)}]", rank)
    print_rank0("=" * 60, rank)

    if args.device == "flagos":
        import torch_flagos

        print_rank0(
            f"Flagos device available: {torch_flagos.flagos.is_available()}", rank
        )
        print_rank0(f"FlagGems registered: {torch_flagos.is_flaggems_enabled()}", rank)
        print_rank0(
            f"Registered ops count: {len(torch_flagos.get_registered_ops())}", rank
        )
    else:
        print_rank0(f"CUDA available: {torch.cuda.is_available()}", rank)
        if torch.cuda.is_available():
            print_rank0(f"CUDA device: {torch.cuda.get_device_name(local_rank)}", rank)

    if args.parallel != "none":
        print_rank0(
            f"World size: {world_size}, rank: {rank}, local_rank: {local_rank}", rank
        )

    # --- Load model ---
    model, tokenizer = load_model(args, device, rank)

    # --- Distributed barrier before wrapping ---
    if args.parallel != "none":
        sync(args)
        t = torch.zeros(1, device=device)
        dist.all_reduce(t)
        sync(args)

    # --- Wrap model ---
    if args.parallel == "ddp":
        model = wrap_ddp(model, args, local_rank, rank)
    elif args.parallel == "fsdp":
        model = wrap_fsdp(model, args, device, rank)

    # --- DataLoader ---
    print_rank0("\n[2] Creating dataset...", rank)
    dataloader, sampler = create_dataloader(args, tokenizer, world_size, rank)
    print_rank0(f"Dataset size: {len(dataloader.dataset)}", rank)
    print_rank0(
        f"Batch size{' per GPU' if args.parallel != 'none' else ''}: {args.batch_size}",
        rank,
    )
    if args.parallel != "none":
        print_rank0(f"Global batch size: {args.batch_size * world_size}", rank)
    print_rank0(f"Sequence length: {args.seq_len}", rank)

    # --- Optimizer ---
    print_rank0("\n[3] Creating optimizer...", rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print_rank0(f"Optimizer: AdamW, lr={args.lr}", rank)

    # --- Training loop ---
    parallel_label = f" {args.parallel.upper()}" if args.parallel != "none" else ""
    print_rank0(
        f"\n[4] Starting{parallel_label} training ({args.steps} steps)...", rank
    )

    total_tokens = 0
    total_loss = 0.0
    step_times = []

    if sampler is not None:
        sampler.set_epoch(0)
    data_iter = iter(dataloader)

    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step + 1)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        sync(args)
        step_start = time.time()

        loss, batch_tokens = train_step(model, batch, device, args)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        sync(args)
        step_time = time.time() - step_start
        step_times.append(step_time)

        total_tokens += batch_tokens
        total_loss += loss.item()

        print_rank0(
            f"  Step {step + 1}/{args.steps}: "
            f"loss={loss.item():.4f}, time={step_time:.2f}s, "
            f"tokens/s={batch_tokens / step_time:.1f}",
            rank,
        )

    # --- Summary ---
    print_summary(args, step_times, total_loss, total_tokens, world_size, rank)
    print_rank0("\nTraining test completed!", rank)

    cleanup(args)


if __name__ == "__main__":
    main()
