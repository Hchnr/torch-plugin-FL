# torch_flagos

FlagGems operators registered as a custom PyTorch device named `flagos`.

## Overview

This package registers FlagGems' high-performance Triton kernels as a custom PyTorch device backend using PyTorch's PrivateUse1 extension mechanism. When you import `torch_flagos`, all FlagGems operators are automatically registered for the `flagos` device - **no need to call `flag_gems.enable()`**.

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0
- CUDA Toolkit
- CMake >= 3.18
- FlagGems (for optimized operators)

### Build from source

```bash
cd torch-plugin-FL
pip install -e . --no-build-isolation
```

## Usage

```python
import torch
import torch_flagos  # Automatically registers FlagGems operators for "flagos" device

# Check device and FlagGems status
print(f"Device available: {torch_flagos.flagos.is_available()}")
print(f"FlagGems enabled: {torch_flagos.is_flaggems_enabled()}")
print(f"Registered ops: {len(torch_flagos.get_registered_ops())}")

# Create tensors on flagos device - operations automatically use FlagGems kernels
x = torch.randn(1000, 1000, device="flagos")
y = torch.randn(1000, 1000, device="flagos")

# All operations use FlagGems kernels (no flag_gems.enable() needed!)
z = x + y
mm_result = torch.mm(x, y)
softmax_result = torch.softmax(x, dim=-1)

# Move tensors between devices
cpu_tensor = torch.randn(3, 3)
flagos_tensor = cpu_tensor.to("flagos")
back_to_cpu = flagos_tensor.cpu()

# Use device context manager
with torch_flagos.flagos.device(0):
    a = torch.randn(10, 10, device="flagos")
```

## Key Difference from `flag_gems.enable()`

| Feature | `flag_gems.enable()` | `torch_flagos` device |
|---------|---------------------|----------------------|
| Activation | Explicit call required | Automatic on import |
| Scope | Global, affects all CUDA tensors | Only `device="flagos"` tensors |
| Isolation | Patches all CUDA ops | Separate device namespace |
| Mixed usage | Hard to mix FlagGems/PyTorch ops | Easy: use `device="cuda"` for PyTorch, `device="flagos"` for FlagGems |

## API Reference

```python
import torch_flagos

# Check if FlagGems is available
torch_flagos.is_flaggems_available()  # -> bool

# Check if FlagGems operators are registered
torch_flagos.is_flaggems_enabled()  # -> bool

# Get list of registered operator names
torch_flagos.get_registered_ops()  # -> List[str]

# Device module (torch_flagos.flagos)
torch_flagos.flagos.is_available()  # -> bool
torch_flagos.flagos.device_count()  # -> int
torch_flagos.flagos.current_device()  # -> int
torch_flagos.flagos.set_device(device_id)
```

## Architecture

The package consists of:

1. **C++ Runtime Layer** (`csrc/runtime/`):
   - Device management (FlagosFunctions.cpp)
   - Memory allocation (FlagosDeviceAllocator.cpp)
   - Random number generation (FlagosGenerator.cpp)
   - PyTorch hooks integration (FlagosHooks.cpp)
   - Device guard (FlagosGuard.cpp)

2. **Operator Registration** (`csrc/aten/`):
   - Basic tensor operations (empty, copy, view, etc.)
   - CPU fallback for unimplemented operations

3. **Python Module** (`torch_flagos/`):
   - Device management API
   - Automatic FlagGems operator registration
   - PyTorch integration via `rename_privateuse1_backend`

## License

Apache License 2.0
