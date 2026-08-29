"""Unit tests for CUDA-first device selection."""

import pytest

from src.common import device
from src.common.device import (
    CPU_COMPUTE_TYPE,
    CUDA_COMPUTE_TYPE,
    DEVICE_AUTO,
    DEVICE_CPU,
    DEVICE_CUDA,
    DeviceUnavailableError,
    resolve,
)


@pytest.fixture(name="with_cuda")
def _with_cuda(monkeypatch):
    monkeypatch.setattr(device, "cuda_is_available", lambda: True)


@pytest.fixture(name="without_cuda")
def _without_cuda(monkeypatch):
    monkeypatch.setattr(device, "cuda_is_available", lambda: False)


class TestAutoPrefersGpu:
    def test_auto_selects_cuda_when_present(self, with_cuda):
        assert resolve(DEVICE_AUTO, "auto").device == DEVICE_CUDA

    def test_auto_reports_that_the_gpu_was_preferred(self, with_cuda):
        assert "preferred" in resolve(DEVICE_AUTO, "auto").reason

    def test_auto_falls_back_to_cpu_when_cuda_is_absent(self, without_cuda):
        """NFR-09: the stack must still run on a laptop with no hardware."""
        assert resolve(DEVICE_AUTO, "auto").device == DEVICE_CPU

    def test_the_fallback_says_why_it_happened(self, without_cuda):
        """Silent CPU fallback on the Jetson is the failure section 9.2 warns
        about, so the reason is always carried."""
        assert "not available" in resolve(DEVICE_AUTO, "auto").reason


class TestExplicitRequests:
    def test_explicit_cuda_is_honoured_when_present(self, with_cuda):
        assert resolve(DEVICE_CUDA, "auto").device == DEVICE_CUDA

    def test_explicit_cuda_raises_rather_than_degrading_silently(self, without_cuda):
        with pytest.raises(DeviceUnavailableError):
            resolve(DEVICE_CUDA, "auto")

    def test_explicit_cpu_is_honoured_even_with_a_gpu_present(self, with_cuda):
        """Reproducing a laptop result on the Jetson needs this."""
        assert resolve(DEVICE_CPU, "auto").device == DEVICE_CPU

    def test_an_unrecognised_device_is_rejected(self):
        with pytest.raises(ValueError):
            resolve("rocm", "auto")


class TestComputeType:
    def test_auto_uses_half_precision_on_the_gpu(self, with_cuda):
        assert resolve(DEVICE_AUTO, "auto").compute_type == CUDA_COMPUTE_TYPE

    def test_auto_uses_int8_on_the_cpu(self, without_cuda):
        assert resolve(DEVICE_AUTO, "auto").compute_type == CPU_COMPUTE_TYPE

    def test_an_explicit_compute_type_overrides_the_default(self, with_cuda):
        assert resolve(DEVICE_AUTO, "int8").compute_type == "int8"

    def test_an_explicit_compute_type_survives_a_cpu_fallback(self, without_cuda):
        assert resolve(DEVICE_AUTO, "float32").compute_type == "float32"


class TestGpuFlag:
    def test_reports_gpu_when_cuda_was_chosen(self, with_cuda):
        assert resolve(DEVICE_AUTO, "auto").is_gpu is True

    def test_reports_not_gpu_when_cpu_was_chosen(self, without_cuda):
        assert resolve(DEVICE_AUTO, "auto").is_gpu is False


class TestProbe:
    def test_a_missing_torch_is_not_an_error(self, monkeypatch):
        """A core-only checkout has no GPU stack; that is normal."""

        def _no_torch(name):
            raise ImportError(name)

        monkeypatch.setattr(device.importlib, "import_module", _no_torch)
        assert device.cuda_is_available() is False

    def test_a_failing_probe_falls_back_rather_than_propagating(self, monkeypatch):
        class _Broken:
            class cuda:  # noqa: N801 - mirrors torch's attribute layout
                @staticmethod
                def is_available():
                    raise RuntimeError("driver mismatch")

        monkeypatch.setattr(device.importlib, "import_module", lambda name: _Broken)
        assert device.cuda_is_available() is False


class TestSelectionIsImmutable:
    def test_a_selection_cannot_be_altered(self, with_cuda):
        selection = resolve(DEVICE_AUTO, "auto")
        with pytest.raises(Exception):
            selection.device = DEVICE_CPU
