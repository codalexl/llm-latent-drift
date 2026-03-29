import torch

from latent_dynamics.steering import steer_toward_reference


def test_steer_toward_reference_reports_delta_norm() -> None:
    hidden = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    reference = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    steered, result = steer_toward_reference(hidden, reference, strength=0.5)
    assert result.applied
    assert result.delta_norm > 0.0
    assert steered.shape == hidden.shape


def test_steer_with_zero_strength_is_noop() -> None:
    hidden = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    reference = torch.tensor([3.0, 4.0], dtype=torch.float32)
    steered, result = steer_toward_reference(hidden, reference, strength=0.0)
    assert not result.applied
    assert result.delta_norm == 0.0
    assert torch.allclose(steered, hidden)
