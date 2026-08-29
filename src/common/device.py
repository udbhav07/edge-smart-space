"""Compute device selection. CUDA first, CPU as the fallback.

The project targets a Jetson AGX Orin, so the GPU is the intended path and
the default asks for it. But NFR-09 requires the full stack to run in
simulation on a development laptop with no hardware attached, and a hard
CUDA default would make that impossible.

Both hold at once through three settings:

``auto``
    Prefer CUDA; fall back to CPU and say so. This is the default, and it is
    what makes the repository GPU-first without stranding anyone.
``cuda``
    Demand CUDA. If it is absent this raises, because an explicit request
    that silently degrades is how a Jetson ends up quietly running inference
    on its CPU at a third of the speed — the exact failure DESIGN.md section
    9.2 warns about.
``cpu``
    Demand CPU, whatever hardware is present. Useful for reproducing a
    laptop result on the Jetson.

Torch is an optional dependency (the ``speech`` extra), so availability is
probed without importing it at module load.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

DEVICE_AUTO = "auto"
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"

VALID_DEVICES = frozenset({DEVICE_AUTO, DEVICE_CUDA, DEVICE_CPU})

COMPUTE_AUTO = "auto"

#: Half precision on the GPU, 8-bit on the CPU. These are the defaults the
#: ``auto`` compute type resolves to; an explicit value always wins.
CUDA_COMPUTE_TYPE = "float16"
CPU_COMPUTE_TYPE = "int8"


class DeviceUnavailableError(RuntimeError):
    """A device was demanded explicitly and is not present."""


@dataclass(frozen=True)
class DeviceSelection:
    """What was chosen, and why.

    ``reason`` is carried so a runner can log the decision. On the Jetson the
    difference between "CUDA, as requested" and "CPU, CUDA not available" is
    the difference between meeting NFR-03 and missing it by a wide margin,
    and that must never be silent.
    """

    device: str
    compute_type: str
    reason: str

    @property
    def is_gpu(self) -> bool:
        return self.device == DEVICE_CUDA


def cuda_is_available() -> bool:
    """Whether a usable CUDA device is present.

    Returns False rather than raising when torch is not installed: a core-only
    checkout has no GPU stack and that is a normal state, not an error.
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        LOGGER.debug("torch is not installed; treating CUDA as unavailable")
        return False

    try:
        return bool(torch.cuda.is_available())
    except (RuntimeError, AssertionError) as exc:
        LOGGER.warning("CUDA probe failed, falling back to CPU: %s", exc)
        return False


def resolve(requested_device: str, requested_compute_type: str) -> DeviceSelection:
    """Turn configured preferences into a concrete device and compute type.

    :raises ValueError: if the requested device is not a recognised name.
    :raises DeviceUnavailableError: if CUDA was demanded and is absent.
    """
    if requested_device not in VALID_DEVICES:
        raise ValueError(
            f"device must be one of {sorted(VALID_DEVICES)}, got {requested_device!r}"
        )

    device, reason = _choose_device(requested_device)
    compute_type = _choose_compute_type(device, requested_compute_type)
    return DeviceSelection(device=device, compute_type=compute_type, reason=reason)


def _choose_device(requested: str) -> tuple[str, str]:
    if requested == DEVICE_CPU:
        return DEVICE_CPU, "CPU requested explicitly"

    if requested == DEVICE_CUDA:
        if not cuda_is_available():
            raise DeviceUnavailableError(
                "CUDA was requested explicitly but is not available. Install a "
                "CUDA-enabled torch, or set the device to 'auto' to fall back "
                "to CPU."
            )
        return DEVICE_CUDA, "CUDA requested explicitly and available"

    if cuda_is_available():
        return DEVICE_CUDA, "CUDA available and preferred"
    return DEVICE_CPU, "CUDA not available, falling back to CPU"


def _choose_compute_type(device: str, requested: str) -> str:
    if requested != COMPUTE_AUTO:
        return requested
    return CUDA_COMPUTE_TYPE if device == DEVICE_CUDA else CPU_COMPUTE_TYPE
