"""
MACA Platform (沐曦) - Basic Operations and Tensor Tests

Usage:
    pytest tests/maca/test_ops.py -v
    LD_LIBRARY_PATH=/opt/maca-3.3.0/tools/cu-bridge/lib:$LD_LIBRARY_PATH pytest tests/maca/test_ops.py -v
"""

import pytest
import torch_flagos  # Must be imported before torch on MACA
import torch

DEVICE = "flagos:0"


def pytest_configure(config):
    if not torch_flagos.flagos.is_available():
        pytest.exit("flagos device is not available.")


@pytest.fixture(scope="session", autouse=True)
def device_info():
    print(f"\nflagos device count={torch_flagos.flagos.device_count()}  "
          f"FlagGems enabled={torch_flagos.is_flaggems_enabled()}  "
          f"registered ops={len(torch_flagos.get_registered_ops())}")


def sync():
    torch_flagos.flagos.synchronize()


# ---------------------------------------------------------------------------
# 1. Tensor creation
# ---------------------------------------------------------------------------

class TestTensorCreation:
    def test_randn(self):
        x = torch.randn(1024, 1024, device=DEVICE)
        assert str(x.device).startswith("flagos")

    def test_zeros(self):
        assert torch.zeros(64, 64, device=DEVICE).sum().item() == 0.0

    def test_ones(self):
        assert torch.ones(64, 64, device=DEVICE).sum().item() == 64 * 64

    def test_arange(self):
        assert torch.arange(10, device=DEVICE)[-1].item() == 9

    def test_full(self):
        assert torch.full((4, 4), 3.14, device=DEVICE)[0, 0].item() == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# 2. Arithmetic operations
# ---------------------------------------------------------------------------

class TestArithmetic:
    @pytest.fixture(autouse=True)
    def tensors(self):
        self.a = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        self.b = torch.tensor([4.0, 5.0, 6.0], device=DEVICE)

    def test_add(self):
        assert torch.allclose(self.a + self.b, torch.tensor([5., 7., 9.], device=DEVICE))

    def test_sub(self):
        assert torch.allclose(self.a - self.b, torch.tensor([-3., -3., -3.], device=DEVICE))

    def test_mul(self):
        assert torch.allclose(self.a * self.b, torch.tensor([4., 10., 18.], device=DEVICE))

    def test_div(self):
        assert torch.allclose(self.a / self.b, torch.tensor([0.25, 0.4, 0.5], device=DEVICE))

    def test_neg(self):
        assert torch.allclose(-self.a, torch.tensor([-1., -2., -3.], device=DEVICE))


# ---------------------------------------------------------------------------
# 3. Matrix operations
# ---------------------------------------------------------------------------

class TestMatrixOps:
    def test_mm(self):
        m = torch.randn(256, 256, device=DEVICE)
        n = torch.randn(256, 256, device=DEVICE)
        result = torch.mm(m, n)
        assert result.shape == (256, 256)
        assert str(result.device).startswith("flagos")

    def test_bmm(self):
        a = torch.randn(8, 64, 128, device=DEVICE)
        b = torch.randn(8, 128, 64, device=DEVICE)
        assert torch.bmm(a, b).shape == (8, 64, 64)

    def test_matmul(self):
        m = torch.randn(256, 256, device=DEVICE)
        n = torch.randn(256, 256, device=DEVICE)
        assert torch.matmul(m, n).shape == (256, 256)


# ---------------------------------------------------------------------------
# 4. Reduction operations
# ---------------------------------------------------------------------------

class TestReductions:
    @pytest.fixture(autouse=True)
    def t(self):
        self.t = torch.randn(128, 128, device=DEVICE)

    def test_sum(self):
        assert self.t.sum().shape == torch.Size([])

    def test_mean(self):
        assert self.t.mean().shape == torch.Size([])

    def test_max(self):
        assert self.t.max().shape == torch.Size([])

    def test_min(self):
        assert self.t.min().shape == torch.Size([])

    def test_std(self):
        assert self.t.std().shape == torch.Size([])

    def test_norm(self):
        assert self.t.norm().shape == torch.Size([])


# ---------------------------------------------------------------------------
# 5. Activation functions (FlagGems kernels)
# ---------------------------------------------------------------------------

class TestActivations:
    @pytest.fixture(autouse=True)
    def inp(self):
        self.inp = torch.randn(64, 64, device=DEVICE)

    def test_relu(self):
        assert torch.nn.functional.relu(self.inp).min().item() >= 0.0

    def test_sigmoid(self):
        s = torch.sigmoid(self.inp)
        assert (s > 0).all() and (s < 1).all()

    def test_tanh(self):
        t = torch.tanh(self.inp)
        assert (t > -1).all() and (t < 1).all()

    def test_softmax(self):
        sm = torch.softmax(self.inp, dim=-1)
        assert torch.allclose(sm.sum(dim=-1), torch.ones(64, device=DEVICE), atol=1e-4)

    def test_log_softmax(self):
        assert (torch.log_softmax(self.inp, dim=-1) <= 0).all()

    def test_silu(self):
        assert torch.nn.functional.silu(self.inp).shape == self.inp.shape

    def test_gelu(self):
        assert torch.nn.functional.gelu(self.inp).shape == self.inp.shape


# ---------------------------------------------------------------------------
# 6. Shape operations
# ---------------------------------------------------------------------------

class TestShapeOps:
    @pytest.fixture(autouse=True)
    def t(self):
        self.t = torch.randn(2, 3, 4, device=DEVICE)

    def test_reshape(self):
        assert self.t.reshape(6, 4).shape == (6, 4)

    def test_view(self):
        assert self.t.view(24).shape == (24,)

    def test_transpose(self):
        assert self.t.transpose(0, 1).shape == (3, 2, 4)

    def test_permute(self):
        assert self.t.permute(2, 0, 1).shape == (4, 2, 3)

    def test_squeeze(self):
        assert torch.randn(1, 3, 1, device=DEVICE).squeeze().shape == (3,)

    def test_unsqueeze(self):
        assert self.t.unsqueeze(0).shape == (1, 2, 3, 4)

    def test_cat(self):
        assert torch.cat([self.t, self.t], dim=0).shape == (4, 3, 4)

    def test_stack(self):
        assert torch.stack([self.t, self.t], dim=0).shape == (2, 2, 3, 4)

    def test_split(self):
        assert len(torch.split(self.t, 1, dim=0)) == 2

    def test_chunk(self):
        assert len(torch.chunk(self.t, 2, dim=0)) == 2


# ---------------------------------------------------------------------------
# 7. Indexing and slicing
# ---------------------------------------------------------------------------

class TestIndexing:
    @pytest.fixture(autouse=True)
    def t(self):
        self.t = torch.arange(24, device=DEVICE).reshape(4, 6).float()

    def test_slice_rows(self):
        assert self.t[1:3].shape == (2, 6)

    def test_slice_cols(self):
        assert self.t[:, 2:5].shape == (4, 3)

    def test_gather(self):
        idx = torch.zeros(4, 1, dtype=torch.long, device=DEVICE)
        assert torch.gather(self.t, 1, idx).shape == (4, 1)

    def test_index_select(self):
        idx = torch.tensor([0, 2], device=DEVICE)
        assert torch.index_select(self.t, 0, idx).shape == (2, 6)


# ---------------------------------------------------------------------------
# 8. Type casting
# ---------------------------------------------------------------------------

class TestTypeCasting:
    @pytest.fixture(autouse=True)
    def f32(self):
        self.f32 = torch.randn(16, device=DEVICE)

    def test_to_float16(self):
        assert self.f32.half().dtype == torch.float16

    def test_to_bfloat16(self):
        assert self.f32.bfloat16().dtype == torch.bfloat16

    def test_to_int32(self):
        assert self.f32.int().dtype == torch.int32

    def test_float16_to_float32(self):
        assert self.f32.half().float().dtype == torch.float32


# ---------------------------------------------------------------------------
# 9. Device transfer
# ---------------------------------------------------------------------------

class TestDeviceTransfer:
    def test_roundtrip(self):
        cpu_t = torch.randn(32, 32)
        flagos_t = cpu_t.to(DEVICE)
        assert torch.allclose(cpu_t, flagos_t.cpu())

    def test_device_type(self):
        assert str(torch.randn(4, device=DEVICE).device).startswith("flagos")


# ---------------------------------------------------------------------------
# 10. Autograd
# ---------------------------------------------------------------------------

class TestAutograd:
    def test_grad_computed(self):
        x = torch.randn(4, 4, device=DEVICE, requires_grad=True)
        (x * x).sum().backward()
        assert x.grad is not None

    def test_grad_shape(self):
        x = torch.randn(4, 4, device=DEVICE, requires_grad=True)
        (x * x).sum().backward()
        assert x.grad.shape == (4, 4)

    def test_grad_values(self):
        x = torch.randn(4, 4, device=DEVICE, requires_grad=True)
        (x * x).sum().backward()
        assert torch.allclose(x.grad, 2 * x.detach(), atol=1e-5)


# ---------------------------------------------------------------------------
# 11. Device context manager
# ---------------------------------------------------------------------------

class TestDeviceContext:
    def test_context_manager(self):
        with torch_flagos.flagos.device(0):
            t = torch.empty(10, 10, device=DEVICE)
        assert str(t.device).startswith("flagos")


# ---------------------------------------------------------------------------
# 12. Synchronization
# ---------------------------------------------------------------------------

class TestSync:
    def test_synchronize(self):
        sync()
