import torch

import torch_flagos._C  # type: ignore[misc]

from . import meta  # noqa: F401


_initialized = False


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = torch.accelerator._get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch_flagos._C._exchangeDevice(self.idx)

    def __exit__(self, type, value, traceback):
        self.idx = torch_flagos._C._set_device(self.prev_idx)
        return False


def is_available():
    return torch_flagos._C._get_device_count() > 0


def device_count() -> int:
    return torch_flagos._C._get_device_count()


def current_device():
    return torch_flagos._C._get_device()


def set_device(device) -> None:
    return torch_flagos._C._set_device(device)


def synchronize(device=None):
    r"""Waits for all operations on the flagos device to complete.

    Args:
        device (torch.device or int, optional): device to synchronize.
            It uses the current device, given by :func:`~torch_flagos.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    if device is not None:
        with torch_flagos.flagos.device(device):
            torch_flagos._C._synchronize()
    else:
        torch_flagos._C._synchronize()


def init():
    _lazy_init()


def is_initialized():
    return _initialized


def _lazy_init():
    global _initialized
    if is_initialized():
        return
    torch_flagos._C._init()
    _initialized = True


from .random import *  # noqa: F403


def get_amp_supported_dtype():
    """Return list of supported dtypes for AMP (Automatic Mixed Precision).

    Required by torch.autocast for custom device backends.
    """
    return [torch.float16, torch.bfloat16]


__all__ = [
    "device",
    "device_count",
    "current_device",
    "set_device",
    "synchronize",
    "initial_seed",
    "is_available",
    "init",
    "is_initialized",
    "random",
    "manual_seed",
    "manual_seed_all",
    "get_rng_state",
    "set_rng_state",
    "get_amp_supported_dtype",
]
