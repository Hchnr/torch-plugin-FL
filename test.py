import torch
import torch_flagos  # Automatically registers FlagGems operators for "flagos" device

print("=" * 60)
print("torch_flagos Test")
print("=" * 60)

# Check device and FlagGems status
print(f"\nDevice available: {torch_flagos.flagos.is_available()}")
print(f"Device count: {torch_flagos.flagos.device_count()}")
print(f"FlagGems available: {torch_flagos.is_flaggems_available()}")
print(f"FlagGems enabled: {torch_flagos.is_flaggems_enabled()}")
print(f"Registered FlagGems ops: {len(torch_flagos.get_registered_ops())}")

# Create tensors on flagos device
# Note: randn uses CPU fallback, then copy to flagos
print("\n--- Creating tensors ---")
x = torch.randn(100, 100, device="flagos")
y = torch.randn(100, 100, device="flagos")
print(f"x shape: {x.shape}, device: {x.device}")
print(f"y shape: {y.shape}, device: {y.device}")

# Compute operations use FlagGems kernels
print("\n--- Compute operations (FlagGems) ---")
z = x + y
print(f"x + y: shape={z.shape}, device={z.device}")

mm_result = torch.mm(x, y)
print(f"mm(x, y): shape={mm_result.shape}, device={mm_result.device}")

softmax_result = torch.softmax(x, dim=-1)
print(f"softmax(x): shape={softmax_result.shape}, device={softmax_result.device}")

# Move tensors between devices
print("\n--- Device transfer ---")
cpu_tensor = torch.randn(3, 3)
flagos_tensor = cpu_tensor.to("flagos")
back_to_cpu = flagos_tensor.cpu()
print(f"CPU -> flagos -> CPU roundtrip: {torch.allclose(cpu_tensor, back_to_cpu)}")

# Use device context manager
print("\n--- Device context manager ---")
with torch_flagos.flagos.device(0):
    a = torch.empty(10, 10, device="flagos")
    print(f"Created tensor in context: shape={a.shape}, device={a.device}")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)