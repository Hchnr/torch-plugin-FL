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

## Testing

Use `tests/test_qwen.sh` to run Qwen3 training tests across multiple configurations (single GPU, DDP, FSDP) with automatic GPU pool scheduling.

```bash
cd torch-plugin-FL

# Default: 3 training steps, auto-detect GPUs
bash tests/test_qwen.sh

# Custom step count
bash tests/test_qwen.sh 10

# Custom steps + log directory
bash tests/test_qwen.sh 3 logs/run1

# Manually specify GPU count
NUM_GPUS=4 bash tests/test_qwen.sh
```

The script runs the following test configurations in parallel (when GPUs are available):

| Task | GPUs | Device | Parallel | Comm |
|------|------|--------|----------|------|
| `cuda_single` | 1 | CUDA | - | - |
| `flagos_single` | 1 | flagos | - | - |
| `cuda_ddp_nccl` | 2 | CUDA | DDP | NCCL |
| `cuda_ddp_flagcx` | 2 | CUDA | DDP | FlagCX |
| `cuda_fsdp_nccl` | 2 | CUDA | FSDP | NCCL |
| `cuda_fsdp_flagcx` | 2 | CUDA | FSDP | FlagCX |
| `flagos_ddp_nccl` | 2 | flagos | DDP | NCCL |
| `flagos_ddp_flagcx` | 2 | flagos | DDP | FlagCX |
| `flagos_fsdp_nccl` | 2 | flagos | FSDP | NCCL |
| `flagos_fsdp_flagcx` | 2 | flagos | FSDP | FlagCX |

Logs are saved to `tests/logs/<timestamp>/` with one file per task. The script prints a pass/fail summary at the end.

You can also run individual configurations directly:

```bash
# Single GPU
python tests/test_qwen3_train.py --device cuda
python tests/test_qwen3_train.py --device flagos

# DDP
torchrun --nproc_per_node=2 tests/test_qwen3_train.py --parallel ddp --comm nccl
torchrun --nproc_per_node=2 tests/test_qwen3_train.py --parallel ddp --comm flagcx

# FSDP
torchrun --nproc_per_node=2 tests/test_qwen3_train.py --parallel fsdp --comm nccl
torchrun --nproc_per_node=2 tests/test_qwen3_train.py --parallel fsdp --comm flagcx
```

## Project Structure

```
torch-plugin-FL/
├── CMakeLists.txt                  # Top-level CMake build
├── setup.py / pyproject.toml       # Python packaging
├── csrc/                           # C++ source code
│   ├── aten/                       # Operator dispatch & registration
│   │   ├── FlagosMinimal.cpp       #   PrivateUse1 operator dispatch registrations
│   │   └── native/                 #   Native implementations
│   │       ├── Minimal.cpp         #     Basic tensor ops (empty, set_, copy, clone, etc.)
│   │       └── Common.h            #     Shared helpers
│   └── runtime/                    # Device runtime layer
│       ├── FlagosFunctions.cpp/h   #   Device management (set/get/exchange device)
│       ├── FlagosDeviceAllocator.* #   GPU memory allocation (wraps cudaMalloc)
│       ├── FlagosHostAllocator.*   #   Pinned host memory allocation
│       ├── FlagosGenerator.*       #   Random number generator
│       ├── FlagosHooks.*           #   PyTorch backend hooks integration
│       ├── FlagosGuard.*           #   Device guard (RAII device switching)
│       └── FlagosException.h       #   Error handling
├── third_party/
│   └── flagos/                     # Low-level device abstraction (CUDA wrappers)
│       ├── include/flagos.h        #   C API (foSetDevice, foMalloc, foStream, etc.)
│       └── csrc/                   #   Implementation (thin wrappers around CUDA APIs)
├── torch_flagos/                   # Python package
│   ├── __init__.py                 #   Entry point: registers FlagGems ops on import
│   ├── integration.py              #   FlagGems operator registration logic
│   ├── distributed.py              #   Distributed training (DDP/FSDP support for flagos)
│   ├── _utils.py                   #   Utility functions
│   ├── flagos/                     #   Device module (torch_flagos.flagos)
│   │   ├── __init__.py             #     Device APIs (set_device, synchronize, etc.)
│   │   ├── random.py               #     RNG APIs
│   │   └── meta.py                 #     Device metadata
│   ├── csrc/                       #   Python C extension
│   │   └── Module.cpp              #     _C module (flagos<->CUDA view conversions)
│   └── lib/                        #   Built shared libraries
└── tests/                          # Test suite
    ├── test_qwen.sh                #   GPU pool scheduler for parallel test execution
    ├── test_qwen3_train.py         #   Qwen3 training test (single/DDP/FSDP)
    ├── test_qwen3_infer.py         #   Qwen3 inference test
    ├── test.py                     #   Basic operator tests
    └── dummy_dataset.py            #   Synthetic dataset for training tests
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
