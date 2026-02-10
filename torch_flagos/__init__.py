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
_autograd_lib = None
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
    # Copy ops - exclude to prevent recursion in _flagos_to_cuda
    # These will use the C++ cpu_fallback which handles flagos correctly
    "copy_", "_to_copy", "contiguous", "clone",
}


def _get_device_index(args, kwargs):
    """Extract flagos device index from arguments."""
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.device.type in ("privateuseone", "flagos"):
            return arg.device.index or 0
    for v in kwargs.values():
        if isinstance(v, torch.Tensor) and v.device.type in ("privateuseone", "flagos"):
            return v.device.index or 0
    return 0


# Cache for CUDA runtime library
_cudart_lib = None
_cudaMemcpy = None


def _get_cudaMemcpy():
    """Get cudaMemcpy function from CUDA runtime library (cached)."""
    global _cudart_lib, _cudaMemcpy
    if _cudaMemcpy is not None:
        return _cudaMemcpy

    import ctypes
    import os

    # Try to load CUDA runtime library
    try:
        _cudart_lib = ctypes.CDLL("libcudart.so")
    except OSError:
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        _cudart_lib = ctypes.CDLL(f"{cuda_home}/lib64/libcudart.so")

    _cudaMemcpy = _cudart_lib.cudaMemcpy
    _cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _cudaMemcpy.restype = ctypes.c_int

    return _cudaMemcpy


# Thread-local set to track tensor IDs currently being converted (to prevent infinite recursion)
import threading
_converting_tensor_ids = threading.local()


class _FlagosToCuda(torch.autograd.Function):
    """Autograd-aware conversion from flagos to CUDA."""
    @staticmethod
    def forward(ctx, t, device_index):
        ctx.device_index = device_index
        import torch_flagos._C as _C
        return _C._flagos_to_cuda_view(t)

    @staticmethod
    def backward(ctx, grad_output):
        import torch_flagos._C as _C
        # grad_output is CUDA tensor, convert to flagos
        flagos_grad = _C._cuda_to_flagos_view(grad_output, ctx.device_index)
        return flagos_grad, None


class _CudaToFlagos(torch.autograd.Function):
    """Autograd-aware conversion from CUDA to flagos."""
    @staticmethod
    def forward(ctx, t, device_index):
        ctx.device_index = device_index
        import torch_flagos._C as _C
        return _C._cuda_to_flagos_view(t, device_index)

    @staticmethod
    def backward(ctx, grad_output):
        import torch_flagos._C as _C
        # grad_output is flagos tensor, convert to CUDA
        cuda_grad = _C._flagos_to_cuda_view(grad_output)
        return cuda_grad, None


def _flagos_to_cuda(t, device_index):
    """
    Convert a flagos tensor to a CUDA tensor using zero-copy view.
    The underlying GPU memory is shared, so no memory copy is made.
    Supports autograd for tensors that require gradients.
    """
    if not isinstance(t, torch.Tensor):
        return t
    if t.device.type not in ("privateuseone", "flagos"):
        return t

    if t.requires_grad:
        return _FlagosToCuda.apply(t, device_index)
    else:
        import torch_flagos._C as _C
        return _C._flagos_to_cuda_view(t)


def _cuda_to_flagos(t, device_index):
    """
    Convert a CUDA tensor to a flagos tensor using zero-copy view.
    The underlying GPU memory is shared, so no memory copy is made.
    Supports autograd for tensors that require gradients.
    """
    if not isinstance(t, torch.Tensor):
        return t
    if t.device.type != "cuda":
        return t

    if t.requires_grad:
        return _CudaToFlagos.apply(t, device_index)
    else:
        import torch_flagos._C as _C
        return _C._cuda_to_flagos_view(t, device_index)


def _make_simple_wrapper(impl_func):
    """
    Create a wrapper that runs FlagGems operation on CUDA internally.

    Always returns flagos tensors to ensure consistent device in autograd graph.
    AutogradPrivateUse1 fallthrough handles backward pass by dispatching to CUDA autograd.
    """
    import functools

    @functools.wraps(impl_func)
    def wrapper(*args, **kwargs):
        device_index = _get_device_index(args, kwargs)

        # Convert flagos tensors to CUDA views for computation
        def to_cuda(t):
            if isinstance(t, torch.Tensor) and t.device.type in ("privateuseone", "flagos"):
                return _flagos_to_cuda(t, device_index)
            return t

        cuda_args = [to_cuda(arg) for arg in args]
        cuda_kwargs = {k: to_cuda(v) for k, v in kwargs.items()}

        # Run FlagGems kernel on CUDA
        with torch.cuda.device(device_index):
            result = impl_func(*cuda_args, **cuda_kwargs)

        # Always convert result back to flagos to keep autograd graph on same device
        def to_flagos(t):
            if isinstance(t, torch.Tensor) and t.device.type == "cuda":
                return _cuda_to_flagos(t, device_index)
            elif isinstance(t, (tuple, list)):
                return type(t)(to_flagos(x) for x in t)
            return t
        return to_flagos(result)

    return wrapper


def _register_flaggems_operators():
    """
    Register FlagGems operators with the PrivateUse1 (flagos) dispatch key.

    Strategy for autograd support:
    1. Register forward ops for PrivateUse1 - converts to CUDA, runs FlagGems, converts back
    2. Let PyTorch's native autograd handle backward by keeping computation on CUDA

    Key insight: FlagGems operations work with autograd on CUDA. By running operations
    on CUDA tensors (that share storage with flagos tensors), we get autograd for free.
    """
    global _flaggems_lib, _autograd_lib, _registered_ops

    try:
        from flag_gems import _FULL_CONFIG
    except ImportError:
        # flag_gems not installed, will use cpu_fallback
        return 0

    _flaggems_lib = torch.library.Library("aten", "IMPL")
    _registered_ops = []

    # Build mapping of backward ops
    backward_ops = {}
    for item in _FULL_CONFIG:
        if len(item) >= 2:
            op_name = item[0]
            if 'backward' in op_name.lower():
                backward_ops[op_name] = item[1]

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
            # Use simple wrapper that converts to CUDA for computation
            # This preserves autograd because CUDA ops support it
            wrapped_func = _make_simple_wrapper(impl_func)
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
