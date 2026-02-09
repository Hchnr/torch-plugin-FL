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


def _flagos_to_cuda(t, device_index):
    """
    Convert a flagos tensor to a CUDA tensor.
    Since flagos and CUDA share the same underlying GPU memory, we create
    a CUDA tensor and copy data using cudaMemcpy.
    """
    if not isinstance(t, torch.Tensor):
        return t
    if t.device.type not in ("privateuseone", "flagos"):
        return t

    # Get the set of tensor IDs being converted (thread-local)
    if not hasattr(_converting_tensor_ids, 'ids'):
        _converting_tensor_ids.ids = set()

    # Check if this specific tensor is already being converted (recursion)
    tensor_id = id(t)
    if tensor_id in _converting_tensor_ids.ids:
        # We're already converting this tensor, skip to prevent infinite recursion
        return t

    # Use the tensor's own device index to ensure correct device mapping
    actual_device_index = t.device.index if t.device.index is not None else device_index

    try:
        _converting_tensor_ids.ids.add(tensor_id)

        # Use _to_copy (implemented in C++) to handle the conversion
        # This properly handles non-contiguous tensors and cross-device copies
        cuda_t = t.to(f"cuda:{actual_device_index}")

        if t.requires_grad and not cuda_t.requires_grad:
            cuda_t.requires_grad_(True)

        return cuda_t
    finally:
        _converting_tensor_ids.ids.discard(tensor_id)


class _CudaToFlagosFunction(torch.autograd.Function):
    """
    Autograd function to convert CUDA tensor to flagos tensor while preserving gradients.
    Uses _to_copy (C++ implementation) for the actual copy.
    """
    @staticmethod
    def forward(ctx, cuda_tensor, device_index):
        import os
        rank = os.environ.get("RANK", "?")
        print(f"[DEBUG rank{rank}] _CudaToFlagosFunction.forward: input device={cuda_tensor.device}, shape={cuda_tensor.shape}", flush=True)
        ctx.device_index = device_index
        ctx.cuda_device = cuda_tensor.device
        # Use detach and _to_copy to avoid autograd issues during conversion
        # The autograd graph is maintained by this Function, not by the underlying copy
        with torch.no_grad():
            print(f"[DEBUG rank{rank}] calling .to(flagos:{device_index})", flush=True)
            flagos_t = cuda_tensor.to(f"flagos:{device_index}")
            print(f"[DEBUG rank{rank}] .to() completed: output device={flagos_t.device}", flush=True)
        return flagos_t

    @staticmethod
    def backward(ctx, grad_output):
        # Convert flagos gradient back to CUDA for backward pass
        if grad_output is None:
            return None, None
        # grad_output is on flagos, convert to CUDA
        cuda_grad = grad_output.to(ctx.cuda_device)
        return cuda_grad, None


def _cuda_to_flagos(t, device_index):
    """
    Convert a CUDA tensor to a flagos tensor.
    Preserves autograd graph using custom autograd function.
    """
    if not isinstance(t, torch.Tensor):
        return t
    if t.device.type != "cuda":
        return t

    if t.requires_grad:
        return _CudaToFlagosFunction.apply(t, device_index)
    else:
        # For non-grad tensors, use direct copy via _to_copy (C++ implementation)
        return t.to(f"flagos:{device_index}")


def _make_autograd_function(op_name, forward_func, backward_func=None):
    """
    Create an autograd Function that properly tracks gradients for flagos tensors.

    The forward pass:
    1. Converts flagos inputs to CUDA views
    2. Runs the FlagGems forward kernel
    3. Saves necessary tensors for backward
    4. Returns result (keeping as CUDA to preserve grad_fn)

    The backward pass:
    1. Uses saved tensors and grad_output
    2. Runs the FlagGems backward kernel (or uses autograd)
    3. Returns gradients
    """

    class FlagosAutogradOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            # Get device index
            device_index = 0
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.device.type in ("privateuseone", "flagos"):
                    device_index = arg.device.index or 0
                    break

            ctx.device_index = device_index
            ctx.op_name = op_name

            # Convert flagos tensors to CUDA views
            cuda_args = tuple(_flagos_to_cuda(arg, device_index) for arg in args)

            # Save input tensors that require grad for backward
            tensors_to_save = [a for a in cuda_args if isinstance(a, torch.Tensor)]
            ctx.save_for_backward(*tensors_to_save)
            ctx.num_inputs = len(args)

            # Run forward with CUDA device context
            with torch.cuda.device(device_index):
                result = forward_func(*cuda_args)

            # Keep result as CUDA tensor to preserve autograd graph
            # The grad_fn will point back to this Function
            return result

        @staticmethod
        def backward(ctx, *grad_outputs):
            device_index = ctx.device_index
            saved_tensors = ctx.saved_tensors

            with torch.cuda.device(device_index):
                if backward_func is not None:
                    # Use explicit backward function
                    grads = backward_func(*grad_outputs, *saved_tensors)
                    if not isinstance(grads, tuple):
                        grads = (grads,)
                else:
                    # Fall back to None gradients (non-differentiable)
                    grads = tuple(None for _ in range(ctx.num_inputs))

            return grads

    return FlagosAutogradOp


def _make_simple_wrapper(impl_func):
    """
    Create a simple wrapper that runs the operation with CUDA device context.

    IMPORTANT: We do NOT convert the result back to flagos because that would
    break the autograd graph. The result will be a CUDA tensor with proper grad_fn.
    This is a trade-off: device type changes from flagos to cuda, but autograd works.
    """
    import functools

    @functools.wraps(impl_func)
    def wrapper(*args, **kwargs):
        device_index = _get_device_index(args, kwargs)

        # Convert flagos tensors to CUDA
        cuda_args = [_flagos_to_cuda(arg, device_index) for arg in args]
        cuda_kwargs = {k: _flagos_to_cuda(v, device_index) for k, v in kwargs.items()}

        # Debug: Check converted tensors for embedding (temporary)
        if impl_func.__name__ == "embedding":
            import os
            rank = os.environ.get("RANK", "?")
            print(f"[DEBUG rank{rank}] embedding: device_index={device_index}", flush=True)
            for i, (orig, cuda) in enumerate(zip(args, cuda_args)):
                if isinstance(orig, torch.Tensor):
                    print(f"[DEBUG rank{rank}]   arg[{i}]: {orig.device} -> {cuda.device}", flush=True)

        # Run with CUDA device context
        with torch.cuda.device(device_index):
            result = impl_func(*cuda_args, **cuda_kwargs)

        # Debug: print result info before conversion
        if impl_func.__name__ == "embedding":
            import os
            rank = os.environ.get("RANK", "?")
            if isinstance(result, torch.Tensor):
                print(f"[DEBUG rank{rank}] embedding result: device={result.device}, shape={result.shape}, requires_grad={result.requires_grad}", flush=True)

        # Convert result back to flagos device
        # This preserves autograd graph using _CudaToFlagosFunction
        def convert_result(r, depth=0):
            if isinstance(r, torch.Tensor) and r.device.type == "cuda":
                if impl_func.__name__ == "embedding":
                    import os
                    rank = os.environ.get("RANK", "?")
                    print(f"[DEBUG rank{rank}] converting tensor: device={r.device}, shape={r.shape}, requires_grad={r.requires_grad}", flush=True)
                converted = _cuda_to_flagos(r, device_index)
                if impl_func.__name__ == "embedding":
                    print(f"[DEBUG rank{rank}] converted to: device={converted.device}", flush=True)
                return converted
            elif isinstance(r, (tuple, list)):
                return type(r)(convert_result(x, depth+1) for x in r)
            else:
                return r

        return convert_result(result)

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
