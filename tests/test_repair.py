"""Does repair_series repair the smoothed steps of a restored series, and only those? A static object restored at every
step, one step blurred by a residual seeing: the power ratio flags that step, the SNR rule with snr_source 'power' weights
its links up, the repair brings it closer to the truth and leaves the other steps as they were."""
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
import torchmfbd
from torchmfbd.noise import step_power_ratio

HERE = Path(__file__).parent


def _config():
    with open(HERE / "joint.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["optimization"]["gpu"] = -1
    cfg.pop("object2", None)
    cfg["regularization"] = {"time": {"lambda": 5e-4}}
    return cfg


def _series(make_synthetic_frames, n_t=6, bad=2, width=2.5, noise=0.002, seed=0):
    """The truth seen sharp at every step but one, which keeps a Gaussian residual blur (a restored product)."""
    obj, _ = make_synthetic_frames(_config(), n_t, 1, 0.0, seed=seed, static=True)
    truth = obj[0].double()
    n = truth.shape[-1]; k = torch.fft.fftfreq(n)
    kk = torch.hypot(*torch.meshgrid(k, k, indexing="ij"))
    g = torch.Generator().manual_seed(seed + 1)
    prod = truth[None].repeat(n_t, 1, 1) + noise * torch.randn(n_t, n, n, generator=g, dtype=torch.float64)
    prod[bad] = torch.fft.ifft2(torch.fft.fft2(truth) * torch.exp(-2 * (np.pi * width * kk)**2)).real + noise * torch.randn(n, n, generator=g, dtype=torch.float64)
    return truth, prod


def test_power_ratio_flags_the_smoothed_step(make_synthetic_frames):
    truth, prod = _series(make_synthetic_frames)
    f = torch.fft.fftfreq(32); rho = torch.hypot(*torch.meshgrid(f, f, indexing="ij")) / 0.335
    r = step_power_ratio(torch.fft.fft2(prod)[None, :, None], rho)[0]
    assert r[2] < 0.5 and torch.all(r[[0, 1, 3, 4, 5]] > 0.8), r.tolist()


def test_interpolation_is_the_default_and_replaces_only_the_weak_step(make_synthetic_frames):
    truth, prod = _series(make_synthetic_frames)
    p = prod.numpy()
    rep, weak = torchmfbd.repair_series(p, _config())
    assert weak.tolist() == [False, False, True, False, False, False]
    assert rep.shape == p.shape
    assert np.array_equal(rep[[0, 1, 3, 4, 5]], p[[0, 1, 3, 4, 5]])
    mu = p.mean(axis=(-1, -2))
    expect = mu[2] * 0.5 * (p[1] / mu[1] + p[3] / mu[3])
    assert np.allclose(rep[2], expect, rtol=0, atol=1e-12)
    t = truth.numpy()
    err = lambda a: np.sqrt(((a - a.mean()) - (t - t.mean()))**2).mean() / t.std()
    assert err(rep[2]) < 0.2 * err(p[2]), (err(p[2]), err(rep[2]))


def test_interpolation_at_the_ends_and_over_a_run():
    n_t, n = 7, 32
    rng = np.random.default_rng(0)
    base = 1.0 + 0.1 * rng.standard_normal((n, n))
    x = np.stack([base * (1 + 0.01 * t) for t in range(n_t)])
    weak = np.array([True, False, False, True, True, False, True])
    out = torchmfbd.repair._interpolate(x, weak)
    mu = x.mean(axis=(-1, -2)); xn = x / mu[:, None, None]
    assert np.allclose(out[0], mu[0] * xn[1]) and np.allclose(out[6], mu[6] * xn[5])           # ends: the nearest good step
    assert np.allclose(out[3], mu[3] * (xn[2] + (xn[5] - xn[2]) / 3))                           # run 3-4: linear between 2 and 5
    assert np.allclose(out[4], mu[4] * (xn[2] + 2 * (xn[5] - xn[2]) / 3))
    assert np.array_equal(out[~weak], x[~weak])


def test_method_is_checked(make_synthetic_frames):
    truth, prod = _series(make_synthetic_frames)
    with pytest.raises(ValueError, match="'interpolate' or 'deconvolve'"):
        torchmfbd.repair_series(prod.numpy(), _config(), method="interp")
    with pytest.raises(ValueError, match="every step is weak"):
        torchmfbd.repair_series(prod.numpy(), _config(), weak_fraction=10.0)


def test_deconvolve_replaces_only_the_weak_step(make_synthetic_frames):
    truth, prod = _series(make_synthetic_frames)
    rep, weak = torchmfbd.repair_series(prod.numpy(), _config(), method="deconvolve")
    assert weak.tolist() == [False, False, True, False, False, False]
    m = rep.shape[-1]; c0 = (32 - m) // 2; p = prod.numpy()[:, c0:c0 + m, c0:c0 + m]; t = truth.numpy()[c0:c0 + m, c0:c0 + m]
    assert np.array_equal(rep[[0, 1, 3, 4, 5]], p[[0, 1, 3, 4, 5]])
    err = lambda a: np.sqrt(((a - a.mean()) - (t - t.mean()))**2).mean() / t.std()
    assert err(rep[2]) < 0.8 * err(p[2]), (err(p[2]), err(rep[2]))


def test_power_source_weights_the_links_of_the_smoothed_step(make_synthetic_frames, monkeypatch):
    truth, prod = _series(make_synthetic_frames)
    seen = []
    solve = torchmfbd.Deconvolution.solve_object_time_evolution_batched

    def spy(self, S, D, Reg_Space, lambda_s, lambda_t):
        seen.append(lambda_t)
        return solve(self, S, D, Reg_Space, lambda_s, lambda_t)

    monkeypatch.setattr(torchmfbd.Deconvolution, "solve_object_time_evolution_batched", spy)
    cfg = _config(); cfg["optimization"]["filter_loss"] = "wiener_time"
    cfg["regularization"]["time"].update(snr_source="power")
    d = torchmfbd.Deconvolution(cfg)
    d.add_frames(prod.float()[None, :, None], sigma=0.01 * torch.ones(1, 6, 1), id_object=0, id_diversity=0, diversity=0.0)
    d.deconvolve(infer_object=False, optimizer="adam", simultaneous_sequences=10, n_iterations=2)
    lam = [float(v) for v in seen[-1][0]]
    assert lam[1] > 1.5 * lam[0] and lam[2] > 1.5 * lam[0] and lam[4] == pytest.approx(lam[0], rel=0.01), lam.tolist()


def test_snr_source_is_checked():
    cfg = _config(); cfg["optimization"]["filter_loss"] = "wiener_time"
    cfg["regularization"]["time"]["snr_source"] = "frame"
    d = torchmfbd.Deconvolution(cfg)
    with pytest.raises(ValueError, match="'frames' or 'power'"):
        d.set_regularizations()
