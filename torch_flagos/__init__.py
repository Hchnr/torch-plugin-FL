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
}


def _get_device_index(args, kwargs):
    """Extract flagos device index from arguments."""
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.device.type == "privateuseone":
            return arg.device.index or 0
    for v in kwargs.values():
        if isinstance(v, torch.Tensor) and v.device.type == "privateuseone":
            return v.device.index or 0
    return 0


def _flagos_to_cuda(t, device_index):
    """
    Convert a flagos tensor to a CUDA tensor view sharing the same storage.
    This preserves the requires_grad attribute for autograd.
    """
    if not isinstance(t, torch.Tensor):
        return t
    if t.device.type != "privateuseone":
        return t

    # Create a CUDA tensor that shares storage with the flagos tensor
    # Use view_as to maintain autograd tracking
    cuda_t = torch.empty(0, device=f"cuda:{device_index}", dtype=t.dtype)
    cuda_t.set_(t.untyped_storage(), t.storage_offset(), t.shape, t.stride())

    if t.requires_grad:
        cuda_t.requires_grad_(True)

    return cuda_t


def _cuda_to_flagos(t, device_index):
    """
    Convert a CUDA tensor to a flagos tensor view sharing the same storage.
    """
    if not isinstance(t, torch.Tensor):
        return t
    if t.device.type != "cuda":
        return t

    flagos_t = torch.empty(0, device=f"flagos:{device_index}", dtype=t.dtype)
    flagos_t.set_(t.untyped_storage(), t.storage_offset(), t.shape, t.stride())

    return flagos_t


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
                if isinstance(arg, torch.Tensor) and arg.device.type == "privateuseone":
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

        # Convert flagos tensors to CUDA views (shares storage)
        cuda_args = [_flagos_to_cuda(arg, device_index) for arg in args]
        cuda_kwargs = {k: _flagos_to_cuda(v, device_index) for k, v in kwargs.items()}

        # Run with CUDA device context
        # Result will be CUDA tensor with proper autograd support
        with torch.cuda.device(device_index):
            result = impl_func(*cuda_args, **cuda_kwargs)

        # DO NOT convert back to flagos - this preserves autograd graph
        # The result will be CUDA tensor(s), which have proper grad_fn
        return result

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
