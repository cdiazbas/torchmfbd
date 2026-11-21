"""Does step_snr measure how far each time step's data are above the noise? It must reproduce the estimate the time
weighting was measured with (signal between 0.1 and 0.3 of the cutoff, noise beyond 1.1, numpy, 2026-11-21), rank
steps by their noise, and refuse images that have no frequencies beyond the cutoff to measure the noise."""
import numpy as np
import pytest
import torch
from torchmfbd.noise import step_snr


def _rho(n, cutoff):
    f = np.fft.fftfreq(n)
    return np.hypot(*np.meshgrid(f, f, indexing="ij")) / cutoff


def _frames(n_steps, n_frames, n, noise, seed=0):
    """A k^-3 scene seen through a Gaussian blur, with white noise of a different level in each step."""
    g = np.random.default_rng(seed)
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij"))
    amp = np.where(k > 0, k, 1.0)**-1.5 * (k > 0)
    obj = np.fft.ifft2(amp * np.exp(2j * np.pi * g.random((n, n)))).real
    obj = 1.0 + 0.1 * obj / obj.std()
    otf = np.exp(-(k / 0.15)**2)
    clean = np.fft.ifft2(np.fft.fft2(obj) * otf).real
    return np.stack([clean + noise[t] * g.standard_normal((n_frames, n, n)) for t in range(n_steps)])


def test_matches_the_measured_estimate():
    fr = _frames(4, 6, 64, [0.01, 0.02, 0.04, 0.08])
    rho = _rho(64, 0.335)
    # the estimate of the night-5 measurements, written in numpy
    x = (fr - fr.mean(axis=(-1, -2), keepdims=True)) * np.outer(np.hanning(64), np.hanning(64))
    P = (np.abs(np.fft.fft2(x))**2).sum(1)
    ref = (P[:, (rho > 0.1) & (rho < 0.3)].mean(-1) - P[:, rho > 1.1].mean(-1)) / P[:, rho > 1.1].mean(-1)
    out = step_snr(torch.tensor(fr), torch.tensor(rho))
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-10)


def test_fourier_input_skips_the_window():
    fr = torch.tensor(_frames(3, 5, 32, [0.01, 0.02, 0.04]))
    rho = torch.tensor(_rho(32, 0.335))
    ft = torch.fft.fft2(fr)
    P = torch.sum(torch.abs(ft)**2, dim=-3)
    ref = (P[..., (rho > 0.1) & (rho < 0.3)].mean(-1) - P[..., rho > 1.1].mean(-1)) / P[..., rho > 1.1].mean(-1)
    torch.testing.assert_close(step_snr(ft, rho, fourier=True), ref.clamp(min=1e-6))


def test_ranks_steps_by_their_noise():
    out = step_snr(torch.tensor(_frames(4, 6, 64, [0.005, 0.01, 0.02, 0.04])), torch.tensor(_rho(64, 0.335)))
    assert torch.all(out[:-1] > out[1:]), out.tolist()
    # white noise: doubling it divides the ratio by about 4 where the signal dominates
    assert 2.5 < float(out[0] / out[1]) < 5.0, out.tolist()


def test_batch_dimensions_are_kept():
    fr = torch.tensor(_frames(6, 4, 32, [0.01] * 6)).reshape(2, 3, 4, 32, 32)
    assert step_snr(fr, torch.tensor(_rho(32, 0.335))).shape == (2, 3)


def test_needs_frequencies_beyond_the_cutoff():
    # sampled so coarsely that even the corners of the frequency grid (sqrt(2) x Nyquist) lie inside 1.1 x the cutoff
    fr = torch.tensor(_frames(2, 4, 32, [0.01, 0.02]))
    with pytest.raises(ValueError, match="beyond 1.1 times"):
        step_snr(fr, torch.tensor(_rho(32, 0.66)))
