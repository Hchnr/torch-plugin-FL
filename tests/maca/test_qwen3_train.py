"""
MACA Platform (沐曦) - Qwen3 End-to-End Training Test

Supports single GPU, DDP, and FSDP with NCCL or FlagCX backends on flagos device.

Usage:
    # Single GPU
    python tests/maca/test_qwen3_train.py

    # DDP with NCCL
    torchrun --nproc_per_node=2 tests/maca/test_qwen3_train.py --parallel ddp --comm nccl

    # DDP with FlagCX
    torchrun --nproc_per_node=2 tests/maca/test_qwen3_train.py --parallel ddp --comm flagcx

    # FSDP with NCCL
    torchrun --nproc_per_node=2 tests/maca/test_qwen3_train.py --parallel fsdp --comm nccl

    # FSDP with FlagCX
    torchrun --nproc_per_node=2 tests/maca/test_qwen3_train.py --parallel fsdp --comm flagcx
"""

import argparse
import functools
import os
import sys
import time

import torch_flagos  # Must be imported before torch on MACA
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from dummy_dataset import DummyTextDataset


def parse_args():
    parser = argparse.ArgumentParser(description="MACA Qwen3 Training Test")
    parser.add_argument("--parallel", choices=["none", "ddp", "fsdp"], default="none")
    parser.add_argument("--comm", choices=["nccl", "flagcx"], default="nccl")
    parser.add_argument("--model", default="/nfs/hcr/models/Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    if args.seq_len is None:
        args.seq_len = 256 if args.parallel != "none" else 1024
    return args


def sync():
    torch_flagos.flagos.synchronize()


def print_rank0(msg, rank):
    if rank == 0:
        print(msg, flush=True)


def setup(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    torch_flagos.flagos.set_device(local_rank)

    if args.parallel == "none":
        return f"flagos:{local_rank}", local_rank, 1, 0

    import torch_flagos.distributed as flagos_dist
    flagos_dist.init_process_group(backend=args.comm)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return f"flagos:{local_rank}", local_rank, world_size, rank


def cleanup(args):
    if args.parallel != "none":
        dist.destroy_process_group()


def load_model(args, device, rank):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print_rank0("\n[1] Loading model...", rank)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    )
    model = model.to(device)
    model.train()

    # Freeze unused parameters
    print_rank0("[1.5] Detecting unused parameters...", rank)
    dummy = torch.randint(0, 1000, (1, 32), device=device)
    with torch.enable_grad():
        out = model(input_ids=dummy, use_cache=False)
        out.logits.sum().backward()
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            param.requires_grad = False
            unused.append(name)
        else:
            param.grad = None
    print_rank0(f"    Frozen {len(unused)} unused parameters", rank)

    sync()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print_rank0(f"    Parameters: {total:.2f}M total, {trainable:.2f}M trainable", rank)
    print_rank0(f"    Load time: {time.time() - t0:.2f}s", rank)
    return model, tokenizer


def wrap_ddp(model, args, rank):
    import torch_flagos.distributed as flagos_dist
    model = flagos_dist.DistributedDataParallel(model)
    print_rank0("    DDP: flagos mode (python_reducer + custom grad hooks)", rank)
    return model


def wrap_fsdp(model, args, device, rank):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer}
    )
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=policy,
        device_id=torch.device(device),
        use_orig_params=True,
    )
    model.train()

    # Validate gradient flow
    print_rank0("[1.5b] Validating FSDP gradient flow...", rank)
    dummy = torch.randint(0, 1000, (1, 32), device=device)
    with torch.enable_grad():
        out = model(input_ids=dummy, use_cache=False)
        out.logits.sum().backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    print_rank0(f"    Parameters without gradient: {len(unused)}", rank)
    model.zero_grad(set_to_none=True)
    print_rank0(f"    FSDP: FULL_SHARD (device=flagos)", rank)
    return model


def create_dataloader(args, tokenizer, world_size, rank):
    dataset = DummyTextDataset(tokenizer, num_samples=100, max_length=args.seq_len)
    sampler = None
    if args.parallel != "none":
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )
    return dataloader, sampler


def train_step(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    return outputs.loss, input_ids.numel()


def print_summary(args, step_times, total_loss, total_tokens, world_size, rank):
    print_rank0("\n" + "=" * 60, rank)
    print_rank0(f"Training Summary (device=flagos, parallel={args.parallel}, comm={args.comm})", rank)
    print_rank0(f"  Steps:        {args.steps}", rank)
    print_rank0(f"  World size:   {world_size}", rank)
    print_rank0(f"  Avg loss:     {total_loss / args.steps:.4f}", rank)
    avg_step = sum(step_times) / len(step_times)
    print_rank0(f"  Avg step:     {avg_step:.2f}s", rank)
    print_rank0(f"  Throughput:   {total_tokens / sum(step_times):.1f} tokens/s", rank)
    print_rank0("=" * 60, rank)


def main():
    args = parse_args()
    device, local_rank, world_size, rank = setup(args)

    print_rank0("=" * 60, rank)
    print_rank0("MACA (flagos) Qwen3 Training Test", rank)
    print_rank0("=" * 60, rank)
    print_rank0(f"device=flagos  parallel={args.parallel}  comm={args.comm}", rank)
    print_rank0(f"batch_size={args.batch_size}  seq_len={args.seq_len}  steps={args.steps}", rank)

    model, tokenizer = load_model(args, device, rank)

    print_rank0(f"\n[2] Wrapping model (parallel={args.parallel})...", rank)
    if args.parallel == "ddp":
        model = wrap_ddp(model, args, rank)
    elif args.parallel == "fsdp":
        model = wrap_fsdp(model, args, device, rank)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    dataloader, sampler = create_dataloader(args, tokenizer, world_size, rank)
    data_iter = iter(dataloader)

    print_rank0(f"\n[3] Training for {args.steps} steps...", rank)
    step_times, total_loss, total_tokens = [], 0.0, 0

    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step + 1)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        sync()
        t0 = time.time()
        loss, batch_tokens = train_step(model, batch, device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        sync()

        elapsed = time.time() - t0
        step_times.append(elapsed)
        total_tokens += batch_tokens
        total_loss += loss.item()
        print_rank0(
            f"  Step {step+1}/{args.steps}: loss={loss.item():.4f}, "
            f"time={elapsed:.2f}s, tokens/s={batch_tokens/elapsed:.1f}",
            rank,
        )

    print_summary(args, step_times, total_loss, total_tokens, world_size, rank)
    print_rank0("Training test completed!", rank)
    cleanup(args)


if __name__ == "__main__":
    main()
