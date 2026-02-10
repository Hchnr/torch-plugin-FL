"""
Qwen3 DDP (Distributed Data Parallel) Training Test - Using torch_flagos

Usage:
    torchrun --nproc_per_node=2 test_qwen3_train_ddp.py

    or with more GPUs:
    torchrun --nproc_per_node=4 test_qwen3_train_ddp.py
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch_flagos  # Automatically registers FlagGems operators to flagos device
import time


def setup_distributed():
    """Initialize distributed training environment"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    # Set both flagos and CUDA device before init_process_group
    # This is important because flagos uses CUDA memory under the hood
    torch_flagos.flagos.set_device(local_rank)
    torch.cuda.set_device(local_rank)

    print(f"[rank{rank}] Setting device to {local_rank}, CUDA current device: {torch.cuda.current_device()}", flush=True)

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

    # Debug: print registered backends
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"[DEBUG] Process group device types: {pg._device_types}", flush=True)
        try:
            pu1_backend = pg._get_backend(torch.device("privateuseone"))
            print(f"[DEBUG] PrivateUse1 backend: {type(pu1_backend).__name__}", flush=True)
        except Exception as e:
            print(f"[DEBUG] Failed to get PrivateUse1 backend: {e}", flush=True)

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
    """Barrier using all_reduce on CUDA device (flagos shares storage with CUDA)"""
    sync()
    # Use CUDA tensor for NCCL communication (flagos and CUDA share the same storage)
    t = torch.zeros(1, device=f"cuda:{local_rank}")
    dist.all_reduce(t)
    sync()


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


def main():
    # Setup distributed environment
    local_rank, world_size, rank = setup_distributed()
    device = f"flagos:{local_rank}"

    print_rank0("=" * 60, rank)
    print_rank0("torch_flagos Qwen3 DDP Training Test", rank)
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
    BATCH_SIZE = 1
    MAX_SEQ_LEN = 256
    NUM_TRAIN_STEPS = 10
    LEARNING_RATE = 1e-5
    GRADIENT_ACCUMULATION_STEPS = 1

    # Load model
    print_rank0("\n[1] Loading model and tokenizer...", rank)
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model to CPU, then move to flagos device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for training to support gradient computation
        device_map="cpu",
        attn_implementation="eager",  # Use eager attention to avoid SDPA device compatibility issues
    )
    model = model.to(device)

    # Explicitly move all buffers to device (some buffers like inv_freq in rotary embeddings
    # may not be properly moved by model.to() for custom devices)
    def move_buffers_to_device(module, device):
        for name, buf in module._buffers.items():
            if buf is not None and buf.device.type != "privateuseone":
                module._buffers[name] = buf.to(device)
        for child in module.children():
            move_buffers_to_device(child, device)

    move_buffers_to_device(model, device)

    model.train()  # Set to training mode

    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    # Synchronize before DDP initialization
    dist_barrier(local_rank)

    # Debug: Check all parameters are on the correct device before DDP
    if rank == 0:
        print("\n[DEBUG] Checking model parameters device before DDP:")
        cpu_params = []
        flagos_params = []
        other_params = []
        for name, param in model.named_parameters():
            device_type = param.device.type
            if device_type == "cpu":
                cpu_params.append(name)
            elif device_type in ("privateuseone", "flagos"):
                flagos_params.append(name)
            else:
                other_params.append((name, device_type))

        print(f"    Parameters on flagos/privateuseone: {len(flagos_params)}")
        print(f"    Parameters on CPU: {len(cpu_params)}")
        if cpu_params:
            print(f"    CPU params (first 5): {cpu_params[:5]}")
        if other_params:
            print(f"    Other devices: {other_params[:5]}")

        # Also check buffers
        cpu_buffers = []
        flagos_buffers = []
        for name, buf in model.named_buffers():
            if buf.device.type == "cpu":
                cpu_buffers.append(name)
            elif buf.device.type in ("privateuseone", "flagos"):
                flagos_buffers.append(name)
        print(f"    Buffers on flagos/privateuseone: {len(flagos_buffers)}")
        print(f"    Buffers on CPU: {len(cpu_buffers)}")
        if cpu_buffers:
            print(f"    CPU buffers (first 5): {cpu_buffers[:5]}")

    # Wrap model with DDP
    # Note: For custom devices like flagos, we don't use device_ids/output_device
    # which are CUDA-specific parameters
    # Use init_sync=False because DDP's internal verification uses distributed ops
    # that don't fully support privateuseone device type yet
    # Use broadcast_buffers=False because buffer sync also has the same issue
    # Since all ranks load from the same checkpoint, params and buffers are already synchronized
    print(f"[rank{rank}] Before DDP init: model device={next(model.parameters()).device}, "
          f"CUDA current device={torch.cuda.current_device()}", flush=True)
    model = DDP(model, init_sync=False, broadcast_buffers=False, gradient_as_bucket_view=True)
    print(f"[rank{rank}] After DDP init successful", flush=True)

    print_rank0(f"Model device: {next(model.parameters()).device}", rank)
    print_rank0(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", rank)
    print_rank0(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M", rank)
    print_rank0(f"Model load time: {time.time() - load_start:.2f}s", rank)

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
    print_rank0(f"\n[4] Starting DDP training ({NUM_TRAIN_STEPS} steps)...", rank)
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

        # Debug info (only print on first step and rank 0)
        if step == 0 and rank == 0:
            print(f"    [DEBUG] outputs.logits.device: {outputs.logits.device}")
            print(f"    [DEBUG] labels.device: {labels.device}")

        # Manually compute cross-entropy loss
        # Both logits and labels should be on flagos device now
        logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]
        shift_logits = logits[..., :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[..., 1:].reshape(-1)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)

        # Debug info
        if step == 0 and rank == 0:
            print(f"    [DEBUG] loss: {loss}")
            print(f"    [DEBUG] loss.device: {loss.device}")
            print(f"    [DEBUG] loss.requires_grad: {loss.requires_grad}")

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
    print_rank0("DDP Training Summary (torch_flagos + FlagGems):", rank)
    print_rank0(f"  World size: {world_size} GPUs", rank)
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

    print_rank0("\nDDP Training test completed!", rank)

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
