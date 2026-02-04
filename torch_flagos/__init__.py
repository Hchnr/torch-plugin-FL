import sys

import torch


if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries


import torch_flagos._C  # type: ignore[misc]
import torch_flagos.flagos


torch.utils.rename_privateuse1_backend("flagos")
torch._register_device_module("flagos", torch_flagos.flagos)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)


# Global library instance to keep registrations alive
_flaggems_lib = None
_registered_ops = []


def _patch_cuda_device_context():
    """
    Monkey-patch torch.cuda.device to handle flagos devices.

    FlagGems internally calls torch_device_fn.device(tensor.device), but when
    tensor.device is 'flagos:0', torch.cuda.device() fails because it expects
    a CUDA device. This patch wraps torch.cuda.device.__init__ to extract just
    the device index from flagos/privateuseone devices.
    """
    _original_cuda_device_init = torch.cuda.device.__init__

    def _patched_cuda_device_init(self, device):
        # Handle flagos/privateuseone devices by extracting just the index
        if hasattr(device, 'type') and hasattr(device, 'index'):
            if device.type in ('privateuseone', 'flagos'):
                device = device.index if device.index is not None else 0
        return _original_cuda_device_init(self, device)

    torch.cuda.device.__init__ = _patched_cuda_device_init


# Patch torch.cuda.device before FlagGems is used
_patch_cuda_device_context()


# Ops that use torch_device_fn.device(device) with explicit device parameter
# These don't work with flagos device and should use cpu_fallback instead
_EXCLUDED_OPS = {
    # Factory functions that take device parameter
    "randn", "randn_like",
    "rand", "rand_like",
    "zeros", "zeros_like",
    "ones", "ones_like",
    "full", "full_like",
    "arange", "arange.start", "arange.start_step",
    "linspace", "logspace",
    "eye", "eye.m",
    "randperm",
    "empty.memory_format",  # Already registered in C++
    "empty_strided",  # Already registered in C++
    # Random ops that use device context
    "uniform_", "normal.float_Tensor", "normal.Tensor_float", "normal.Tensor_tensor",
    "exponential_", "multinomial",
}


def _make_flagos_wrapper(impl_func):
    """
    Create a wrapper that sets up CUDA device context using device index
    before calling the FlagGems implementation.

    FlagGems ops use torch.cuda.device(tensor.device) internally, but flagos
    tensors have device type "privateuseone" which isn't recognized by CUDA.
    This wrapper extracts the device index and sets CUDA context properly.
    """
    import functools

    @functools.wraps(impl_func)
    def wrapper(*args, **kwargs):
        # Find the first tensor argument to get device index
        device_index = None
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device.type == "privateuseone":
                device_index = arg.device.index
                break
        if device_index is None:
            for v in kwargs.values():
                if isinstance(v, torch.Tensor) and v.device.type == "privateuseone":
                    device_index = v.device.index
                    break

        # Set CUDA device context using index (flagos uses same device indices as CUDA)
        if device_index is not None:
            with torch.cuda.device(device_index):
                return impl_func(*args, **kwargs)
        else:
            return impl_func(*args, **kwargs)

    return wrapper


def _register_flaggems_operators():
    """Register FlagGems operators with the PrivateUse1 (flagos) dispatch key."""
    global _flaggems_lib, _registered_ops

    try:
        from flag_gems import _FULL_CONFIG
    except ImportError:
        # flag_gems not installed, will use cpu_fallback
        return 0

    _flaggems_lib = torch.library.Library("aten", "IMPL")
    _registered_ops = []

    for item in _FULL_CONFIG:
        if len(item) < 2:
            continue

        op_name = item[0]
        impl_func = item[1]

        # Skip excluded ops - they will use cpu_fallback
        if op_name in _EXCLUDED_OPS:
            continue

        # Check version conditions if present
        if len(item) > 2:
            condition = item[2]
            if callable(condition) and not condition():
                continue

        try:
            # Wrap the implementation to handle flagos device context
            wrapped_func = _make_flagos_wrapper(impl_func)
            _flaggems_lib.impl(op_name, wrapped_func, "PrivateUse1")
            _registered_ops.append(op_name)
        except Exception:
            # Some operators may already be registered or have incompatible signatures
            pass

    return len(_registered_ops)


def get_registered_ops():
    """Return list of registered FlagGems operators for flagos device."""
    return list(_registered_ops)


def is_flaggems_enabled():
    """Check if FlagGems operators are registered for flagos device."""
    return len(_registered_ops) > 0


# Auto-register FlagGems operators on import
_register_flaggems_operators()


# Re-export integration utilities
from torch_flagos.integration import (
    is_flaggems_available,
    enable_flaggems_for_flagos,
    use_flaggems,
)

__all__ = [
    "flagos",
    "get_registered_ops",
    "is_flaggems_enabled",
    "is_flaggems_available",
    "enable_flaggems_for_flagos",
    "use_flaggems",
]
