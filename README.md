
A PyTorch plugin built on FlagOS to provide a unified and efficient multi-chip support.

## Overview

This package registers FlagGems' high-performance Triton kernels as a custom PyTorch device backend using PyTorch's PrivateUse1 extension mechanism. When you import `torch_flagos`, all FlagGems operators are automatically registered for the `flagos` device.

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
# For CUDA platform
pip install -e . --no-build-isolation

# For MACA platform
GPU_PLATFORM=muxi pip install -e . --no-build-isolation
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

The test suite is split into integration tests (CI) and manual scripts (interactive exploration).

```
tests/
├── common/              # Shared utilities (DummyTextDataset)
├── integration/         # CI integration tests — same scripts for both platforms
└── manual/              # Manual scripts for interactive exploration
```

### Integration tests (`tests/integration/`)

All three scripts accept a `--device` flag (`cuda` or `flagos`) and are otherwise identical — this is the core goal of `torch_flagos`.

**Basic ops and tensor tests (pytest):**

```bash
pytest tests/integration/test_ops.py -v --device cuda    # CUDA
pytest tests/integration/test_ops.py -v --device flagos  # MACA
```

**Qwen3 inference (pytest):**

```bash
pytest tests/integration/test_qwen3_infer.py -v -s --device cuda
pytest tests/integration/test_qwen3_infer.py -v -s --device flagos
pytest tests/integration/test_qwen3_infer.py -v -s --device cuda --model /path/to/Qwen3-0.6B --max-new-tokens 64
```

**Qwen3 training (pytest, single GPU):**

```bash
pytest tests/integration/test_qwen3_train.py -v -s --device cuda  --steps 10
pytest tests/integration/test_qwen3_train.py -v -s --device flagos --steps 10
```

## MACA Specific
Add the MACA cu-bridge library path to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/opt/maca-3.3.0/tools/cu-bridge/lib:$LD_LIBRARY_PATH
```

Install Triton (versions for MACA).

Install Torch-plugin-FL.
```bash
GPU_PLATFORM=muxi pip install -e . --no-build-isolation -v
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
    ├── common/                     #   Shared utilities
    │   └── dummy_dataset.py        #     Synthetic dataset for training tests
    ├── integration/                #   CI integration tests (cuda + flagos, same scripts)
    │   ├── test_ops.py             #     Basic ops and tensor tests (pytest, --device cuda|flagos)
    │   ├── test_qwen3_infer.py     #     Qwen3 end-to-end inference (--device cuda|flagos)
    │   └── test_qwen3_train.py     #     Qwen3 end-to-end training (--device cuda|flagos, single/DDP/FSDP)
    └── manual/                     #   Manual scripts for interactive exploration
        ├── test.py                 #     flagos device + FlagGems smoke test
        ├── test_qwen3_infer.py     #     Qwen3 inference (flagos)
        ├── test_qwen3_train.py     #     Qwen3 training (cuda/flagos, single/DDP/FSDP)
        ├── test_qwen.sh            #     GPU pool scheduler (parallel multi-config run)
        ├── dummy_dataset.py        #     Synthetic dataset
        └── muxi/
            └── test_cuda_api.c     #     Low-level MACA CUDA API test (C)
```

## License

Apache License 2.0
