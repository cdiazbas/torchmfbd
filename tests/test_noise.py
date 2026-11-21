"""Does compute_noise give the estimate of noise_svd.noise_estimation, without its float32 rounding? Against the numpy
estimate computed in float64 it must agree to rounding; it must recover a known white noise where the noise dominates
the high singular values, and not depend on which other frames come with a frame."""
import numpy as np
import torch
from torchmfbd.noise import compute_noise
from torchmfbd.noise_svd import noise_estimation


def _frames(n_seq, n_frames, n, sigma, seed=0, square=True):
    g = torch.Generator().manual_seed(seed)
    k = torch.hypot(*torch.meshgrid(torch.fft.fftfreq(n), torch.fft.fftfreq(n), indexing="ij"))
    amp = torch.where(k > 0, k, torch.ones_like(k))**-1.5 * (k > 0)
    obj = torch.fft.ifft2(amp * torch.exp(2j * np.pi * torch.rand(n, n, generator=g))).real
    obj = 1 + 0.1 * obj / obj.std()
    fr = obj + torch.as_tensor(sigma).reshape(-1, 1, 1, 1) * torch.randn(n_seq, n_frames, n, n, generator=g)
    return fr if square else fr[..., : n // 2 + 3]


def test_matches_the_numpy_estimate_in_float64():
    for square in (True, False):                                      # m > n and m < n branches
        fr = _frames(4, 3, 64, [0.01, 0.02, 0.05, 0.1], square=square).double()
        ref = noise_estimation(fr.numpy())                            # float64 input: no float32 rounding
        np.testing.assert_allclose(compute_noise(fr).double().numpy(), ref, rtol=1e-6)


def test_recovers_a_dominant_white_noise():
    sig = torch.tensor([0.1, 0.2, 0.4])
    est = compute_noise(_frames(3, 4, 64, sig).float())
    torch.testing.assert_close(est.mean(dim=1), sig, rtol=0.1, atol=0)


def test_a_frame_does_not_depend_on_its_company():
    fr = _frames(6, 4, 32, torch.linspace(0.01, 0.1, 6)).float()
    torch.testing.assert_close(compute_noise(fr)[2:5], compute_noise(fr[2:5]), rtol=0, atol=0)


def test_shape_dtype_device():
    out = compute_noise(_frames(2, 5, 32, [0.02, 0.02]).float())
    assert out.shape == (2, 5) and out.dtype == torch.float32
