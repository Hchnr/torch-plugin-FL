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
from torch.utils.data import DataLoader, DistributedSampler
import torch_flagos  # Automatically registers FlagGems operators to flagos device
import time
from dummy_dataset import DummyTextDataset


def _patch_dist_collectives():
    """Patch torch.distributed collectives to transparently handle flagos tensors.

    After patching, dist.all_reduce(flagos_tensor) works directly without
    any manual _flagos_to_cuda_view conversion in user code.
    """
    import torch_flagos._C as _C

    def _ensure_cuda(tensor):
        if tensor.device.type in ("privateuseone", "flagos"):
            return _C._flagos_to_cuda_view(tensor)
        return tensor

    _orig_all_reduce = dist.all_reduce
    def _all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        return _orig_all_reduce(_ensure_cuda(tensor), op=op, group=group, async_op=async_op)
    dist.all_reduce = _all_reduce

    _orig_broadcast = dist.broadcast
    def _broadcast(tensor, src, group=None, async_op=False):
        return _orig_broadcast(_ensure_cuda(tensor), src=src, group=group, async_op=async_op)
    dist.broadcast = _broadcast

    _orig_reduce = dist.reduce
    def _reduce(tensor, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
        return _orig_reduce(_ensure_cuda(tensor), dst=dst, op=op, group=group, async_op=async_op)
    dist.reduce = _reduce

    _orig_all_gather_into_tensor = dist.all_gather_into_tensor
    def _all_gather_into_tensor(output, input, group=None, async_op=False):
        return _orig_all_gather_into_tensor(
            _ensure_cuda(output), _ensure_cuda(input), group=group, async_op=async_op,
        )
    dist.all_gather_into_tensor = _all_gather_into_tensor

    _orig_reduce_scatter_tensor = dist.reduce_scatter_tensor
    def _reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM, group=None, async_op=False):
        return _orig_reduce_scatter_tensor(
            _ensure_cuda(output), _ensure_cuda(input), op=op, group=group, async_op=async_op,
        )
    dist.reduce_scatter_tensor = _reduce_scatter_tensor


def setup_distributed():
    """Initialize distributed training environment"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    # Set both flagos and CUDA device before init_process_group
    # This is important because flagos uses CUDA memory under the hood
    torch_flagos.flagos.set_device(local_rank)
    torch.cuda.set_device(local_rank)

    from torch.distributed.distributed_c10d import ProcessGroup

    # Initialize with NCCL backend for CUDA
    dist.init_process_group(backend="nccl")

    # Get the default process group and manually register NCCL backend for privateuseone
    pg = dist.distributed_c10d._get_default_group()

    # Get the NCCL backend that was created for CUDA
    nccl_backend = pg._get_backend(torch.device("cuda"))

    # Register the same NCCL backend for privateuseone device type
    # This allows distributed ops on flagos tensors to use NCCL
    pg._register_backend(
        torch.device("privateuseone"),
        ProcessGroup.BackendType.NCCL,
        nccl_backend
    )

    # Patch dist collectives to transparently handle flagos tensors
    _patch_dist_collectives()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

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


def dist_barrier(local_rank):
    """Barrier using all_reduce on flagos device"""
    sync()
    t = torch.zeros(1, device=f"flagos:{local_rank}")
    dist.all_reduce(t)
    sync()


def move_buffers_to_device(module, device):
    """Explicitly move all buffers to device.

    Some buffers like inv_freq in rotary embeddings may not be properly
    moved by model.to() for custom devices.
    """
    for name, buf in module._buffers.items():
        if buf is not None and buf.device.type != "privateuseone":
            module._buffers[name] = buf.to(device)
    for child in module.children():
        move_buffers_to_device(child, device)


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
    MAX_SEQ_LEN = 256
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

    # Load model to CPU first, then move to flagos device
    # We must move to device BEFORE FSDP wrapping because FSDP's internal
    # _move_states_to_device uses param.data = param.to(device) which fails
    # with "incompatible tensor type" for CPU -> PrivateUse1 transitions.
    # By moving first, FSDP sees params already on device and skips the move.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model = model.to(device)

    # Explicitly move all buffers to device (some buffers like inv_freq
    # may not be properly moved by model.to() for custom devices)
    move_buffers_to_device(model, device)

    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    # Synchronize before FSDP initialization
    sync()
    dist_barrier(local_rank)

    # Get FSDP wrap policy
    auto_wrap_policy = get_fsdp_wrap_policy(model)

    # Configure mixed precision (optional, set to None for full float32)
    mixed_precision_policy = None

    # Configure CPU offload (optional)
    cpu_offload = CPUOffload(offload_params=True) if USE_CPU_OFFLOAD else None

    # Wrap model with FSDP
    # Model is already on flagos device; pass device_id so FSDP knows
    # the compute device (it won't move since params aren't on CPU)
    model = FSDP(
        model,
        sharding_strategy=SHARDING_STRATEGY,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        cpu_offload=cpu_offload,
        device_id=torch.device(device),
        use_orig_params=True,
    )

    model.train()

    # Ensure all buffers are on the correct device after FSDP wrapping
    move_buffers_to_device(model, device)

    print_rank0(f"FSDP Sharding Strategy: {SHARDING_STRATEGY}", rank)
    print_rank0(f"CPU Offload: {USE_CPU_OFFLOAD}", rank)
    print_rank0(f"Model load time: {time.time() - load_start:.2f}s", rank)

    # Print FSDP sharded parameter info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"Total parameters (sharded): {total_params / 1e6:.2f}M", rank)
    print_rank0(f"Trainable parameters (sharded): {trainable_params / 1e6:.2f}M", rank)

    # Detect unused parameters via a dummy forward+backward pass
    # This serves as validation that all parameters receive gradients
    print_rank0("\n[1.5] Detecting unused parameters (validation)...", rank)
    dummy_input = torch.randint(0, 1000, (1, 32), device=device)
    with torch.enable_grad():
        dummy_output = model(input_ids=dummy_input, attention_mask=None, labels=None, use_cache=False)
        dummy_loss = dummy_output.logits.sum()
        dummy_loss.backward()

    # Check which parameters didn't receive gradients
    unused_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            unused_params.append(name)
        elif param.grad is not None:
            param.grad = None  # Clear gradient for actual training

    print_rank0(f"    Parameters without gradient: {len(unused_params)}", rank)
    if rank == 0 and unused_params:
        for name in unused_params[:5]:
            print(f"      - {name}")
        if len(unused_params) > 5:
            print(f"      ... and {len(unused_params) - 5} more")

    # Zero out gradients after detection
    model.zero_grad(set_to_none=True)

    sync()
    dist_barrier(local_rank)

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
        # Note: attention_mask not used due to device compatibility issues with transformers' masking_utils
        labels = batch["labels"].to(device)

        sync()
        step_start = time.time()

        # Forward pass
        # Note: We don't pass attention_mask to avoid device compatibility issues with
        # transformers' masking_utils which uses vmap internally and creates tensors on
        # incompatible devices. Since we use padding="max_length" and drop_last=True,
        # all sequences are the same length and the causal mask is sufficient.
        # Also, we don't pass labels to avoid transformers internal contiguous() calls.
        outputs = model(
            input_ids=input_ids,
            attention_mask=None,  # Skip attention_mask to avoid masking_utils device issues
            labels=None,  # Don't compute loss internally, we'll do it manually
            use_cache=False,  # Disable KV cache for training
        )

        # Manually compute cross-entropy loss
        # Both logits and labels should be on flagos device now
        logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]
        shift_logits = logits[..., :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[..., 1:].reshape(-1)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)

        # Backward pass
        # FSDP handles gradient synchronization internally via reduce_scatter
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
