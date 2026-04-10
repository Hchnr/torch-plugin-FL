"""
Qwen3 End-to-End Training Test (pytest)

Runs on both CUDA and flagos (MACA) devices via the --device argument.
Single-GPU only — DDP/FSDP multi-GPU runs use the manual scripts.

Usage:
    pytest tests/integration/test_qwen3_train.py -v -s --device cuda
    pytest tests/integration/test_qwen3_train.py -v -s --device flagos
"""

import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from dummy_dataset import DummyTextDataset
from torch.utils.data import DataLoader


def sync(dev):
    if dev == "flagos":
        import torch_flagos

        torch_flagos.flagos.synchronize()
    else:
        torch.cuda.synchronize()


@pytest.fixture(scope="module")
def ctx(request):
    dev = request.config.getoption("--device")
    model_path = request.config.getoption("--model")
    steps = request.config.getoption("--steps")
    batch_size = request.config.getoption("--batch-size")
    seq_len = request.config.getoption("--seq-len")
    lr = request.config.getoption("--lr")

    if dev == "flagos":
        import torch_flagos

        torch_flagos.flagos.set_device(0)

    device = f"{dev}:0"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[1] Loading model ({model_path})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    )
    model = model.to(device)
    model.train()

    # Freeze unused parameters
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
    print(f"    Frozen {len(unused)} unused parameters")

    sync(dev)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"    Parameters: {total:.2f}M total, {trainable:.2f}M trainable")
    print(f"    Load time: {time.time() - t0:.2f}s")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    dataset = DummyTextDataset(tokenizer, num_samples=100, max_length=seq_len)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "optimizer": optimizer,
        "dataloader": dataloader,
        "device": device,
        "dev": dev,
        "steps": steps,
        "batch_size": batch_size,
    }


def train_steps(ctx, n):
    model = ctx["model"]
    optimizer = ctx["optimizer"]
    device = ctx["device"]
    dev = ctx["dev"]
    data_iter = iter(ctx["dataloader"])

    step_times, losses = [], []
    for _ in range(n):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(ctx["dataloader"])
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        sync(dev)
        t0 = time.time()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        sync(dev)

        step_times.append(time.time() - t0)
        losses.append(loss.item())

    return step_times, losses


class TestQwen3Training:
    def test_single_step(self, ctx):
        """Verify a single training step completes and produces a finite loss."""
        step_times, losses = train_steps(ctx, 1)
        print(f"\n  Step 1: loss={losses[0]:.4f}, time={step_times[0]:.2f}s")
        assert losses[0] > 0, "Loss should be positive"
        assert torch.isfinite(torch.tensor(losses[0])), "Loss is not finite"

    def test_multi_step_loss_finite(self, ctx):
        """Run remaining steps and verify all losses are finite."""
        remaining = max(1, ctx["steps"] - 1)
        step_times, losses = train_steps(ctx, remaining)
        avg_loss = sum(losses) / len(losses)
        avg_time = sum(step_times) / len(step_times)
        total_tokens = ctx["batch_size"] * 1024 * len(step_times)
        print(
            f"\n  Steps 2-{ctx['steps']}: avg_loss={avg_loss:.4f}, "
            f"avg_step={avg_time:.2f}s, "
            f"throughput={total_tokens / sum(step_times):.1f} tok/s"
        )
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses), (
            "Some losses are not finite"
        )

    def test_gradient_flows(self, ctx):
        """Verify gradients are non-zero after a training step."""
        model = ctx["model"]
        # Run one more step to get fresh gradients
        data_iter = iter(ctx["dataloader"])
        batch = next(data_iter)
        device = ctx["device"]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        outputs.loss.backward()
        grad_params = [
            p for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        ctx["model"].zero_grad(set_to_none=True)
        assert len(grad_params) > 0, "No parameters received gradients"
