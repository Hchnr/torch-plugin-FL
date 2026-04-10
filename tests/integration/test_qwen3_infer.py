"""
Qwen3 End-to-End Inference Test (pytest)

Runs on both CUDA and flagos (MACA) devices via the --device argument.

Usage:
    pytest tests/integration/test_qwen3_infer.py -v -s --device cuda
    pytest tests/integration/test_qwen3_infer.py -v -s --device flagos
"""

import time
import pytest
import torch


@pytest.fixture(scope="module")
def ctx(request):
    dev = request.config.getoption("--device")
    model_path = request.config.getoption("--model")
    max_new_tokens = request.config.getoption("--max-new-tokens")

    if dev == "flagos":
        import torch_flagos

        print(
            f"\nflagos device count={torch_flagos.flagos.device_count()}  "
            f"FlagGems enabled={torch_flagos.is_flaggems_enabled()}  "
            f"registered ops={len(torch_flagos.get_registered_ops())}"
        )
    else:
        print(f"\nCUDA device: {torch.cuda.get_device_name(0)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    device = f"{dev}:0"
    model = model.to(device)
    model.eval()
    print(
        f"Model device: {next(model.parameters()).device}  load time: {time.time() - t0:.2f}s"
    )

    text = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Give me a short introduction to large language model.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    print(f"Input tokens: {inputs.input_ids.shape[1]}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "inputs": inputs,
        "device": dev,
        "max_new_tokens": max_new_tokens,
    }


def sync(dev):
    if dev == "flagos":
        import torch_flagos

        torch_flagos.flagos.synchronize()
    else:
        torch.cuda.synchronize()


def run_inference(ctx):
    model, inputs, dev = ctx["model"], ctx["inputs"], ctx["device"]
    sync(dev)
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=ctx["max_new_tokens"])
    sync(dev)
    elapsed = time.time() - t0
    new_tokens = output.shape[1] - inputs.input_ids.shape[1]
    return output, elapsed, new_tokens


class TestQwen3Inference:
    def test_first_inference(self, ctx):
        """First run — may include kernel compilation."""
        output, elapsed, new_tokens = run_inference(ctx)
        print(
            f"\n  First: {elapsed:.2f}s, {new_tokens} tokens, {new_tokens / elapsed:.2f} tok/s"
        )
        assert new_tokens > 0, "No tokens generated on first run"

    def test_second_inference(self, ctx):
        """Second run — kernel cache warm."""
        output, elapsed, new_tokens = run_inference(ctx)
        print(
            f"\n  Second: {elapsed:.2f}s, {new_tokens} tokens, {new_tokens / elapsed:.2f} tok/s"
        )
        assert new_tokens > 0

    def test_third_inference(self, ctx):
        output, elapsed, new_tokens = run_inference(ctx)
        print(
            f"\n  Third: {elapsed:.2f}s, {new_tokens} tokens, {new_tokens / elapsed:.2f} tok/s"
        )
        assert new_tokens > 0

    def test_fourth_inference_and_output(self, ctx):
        """Fourth run — verify non-empty decoded text."""
        output, elapsed, new_tokens = run_inference(ctx)
        print(
            f"\n  Fourth: {elapsed:.2f}s, {new_tokens} tokens, {new_tokens / elapsed:.2f} tok/s"
        )
        decoded = ctx["tokenizer"].decode(
            output[0][ctx["inputs"].input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"\nGenerated output:\n{decoded}")
        assert new_tokens > 0
        assert len(decoded.strip()) > 0, "Generated text is empty"
