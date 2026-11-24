import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import torchmfbd


def _redux_reference(P2, Q, s2, rho, cutoff=0.9):
    """
    A line-by-line transcription of MOMFBD's ScharmerFilter (redux, src/lib/image/utils.cpp) on the centred grid, with
    its own loops: smooth(pt, 1, 1), the gain from the ratio, the f / -f average, the 0.2 threshold, the 4-connected
    region from the centre that keeps its values, smooth(rr, 4, 4), and the cutoff at `cutoff` of the diffraction limit;
    then, as the library and unlike redux, the zero frequency back to 1.
    """
    ny, nx = Q.shape

    def smooth(a, n):
        out = np.zeros_like(a)
        for y in range(ny):
            yl, yh = max(y - n, 0), min(y + n, ny)
            for x in range(nx):
                xl, xh = max(x - n, 0), min(x + n, nx)
                out[y, x] = a[yl:yh, xl:xh].mean()
        return out

    pt = smooth(np.where(Q > 0, P2 / np.where(Q > 0, Q, 1.0), 0.0), 1)
    pt = np.where(pt > 0, s2 / np.where(pt > 0, pt, 1.0), np.inf)
    rr = np.zeros_like(pt)
    for y in range(ny):
        for x in range(nx):
            rr[y, x] = max(1.0 - 0.5 * (pt[(ny - y) % ny, (nx - x) % nx] + pt[y, x]), 0.0)
    rr[rr < 0.2] = 0.0
    rr[rr > 1.0] = 1.0
    rr[ny // 2, nx // 2] = 1.0
    keep = np.zeros_like(rr, dtype=bool)
    stack = [(ny // 2, nx // 2)]
    keep[ny // 2, nx // 2] = True
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= yy < ny and 0 <= xx < nx and not keep[yy, xx] and rr[yy, xx] > 0:
                keep[yy, xx] = True
                stack.append((yy, xx))
    rr = smooth(rr * keep, 4)
    rr[ny // 2, nx // 2] = 1.0          # the library keeps the mean (redux leaves the smoothed value)
    rr[rho > cutoff] = 0.0
    return rr


def _sums(n=32, n_seq=2, seed=1):
    # sum|S|^2 falling with frequency and sum S*D with a signal that falls faster than the noise: a central region of
    # signal surrounded by noise, as a restored patch
    gen = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    r = np.hypot(*np.meshgrid(f, f, indexing="ij")) / 0.4
    Q = np.exp(-(r / 0.5)**2)[None].repeat(n_seq, 0) + 1e-3
    noise = (gen.normal(size=(n_seq, n, n)) + 1j * gen.normal(size=(n_seq, n, n))) * np.sqrt(Q)
    P = 20.0 * np.exp(-(r / 0.25)**2) * Q + noise
    return torch.tensor(Q), torch.tensor(P), r / 1.0


def test_scharmer_filter_matches_redux():
    # The library's filter equals a direct transcription of MOMFBD's (redux) ScharmerFilter
    Q, P, r = _sums()
    s2 = 1.0
    dummy = types.SimpleNamespace(rho=[torch.tensor(r)], _box_mean=torchmfbd.Deconvolution._box_mean)
    gain = torchmfbd.Deconvolution.scharmer_filter(dummy, Q, P, s2, 0).numpy()
    rho_c = np.fft.fftshift(r)
    for ib in range(Q.shape[0]):
        Qc = np.fft.fftshift(Q[ib].numpy())
        P2c = np.fft.fftshift(np.abs(P[ib].numpy())**2)
        ref = _redux_reference(P2c, Qc, s2, rho_c)
        assert np.allclose(np.fft.fftshift(gain[ib]), ref, atol=1e-6), f"sequence {ib}"


def test_scharmer_filter_properties():
    # A continuous gain in [0, 1], 1 at zero frequency, 0 beyond 0.9 of the cutoff, below the binary mask's 1 somewhere
    Q, P, r = _sums()
    dummy = types.SimpleNamespace(rho=[torch.tensor(r)], _box_mean=torchmfbd.Deconvolution._box_mean)
    gain = torchmfbd.Deconvolution.scharmer_filter(dummy, Q, P, 1.0, 0).numpy()
    assert gain.min() >= 0.0 and gain.max() <= 1.0 + 1e-6
    assert np.all(gain[:, r > 0.9] == 0.0)
    assert np.any((gain > 0.05) & (gain < 0.95)), "the gain should be continuous, not a mask"
    assert np.all(gain[:, 0, 0] == 1.0), "the object keeps its mean"


def test_scharmer_variants_share_the_fit(make_synthetic_frames):
    # The two scharmer filters shape only the restored object: the fit (modes) is the same, the objects differ
    config_path = Path(__file__).parent / 'joint.yaml'
    obj, frames = make_synthetic_frames(config_path, 4, 5, 0.005, seed=0)
    out = {}
    for filt in ('scharmer', 'scharmer_binary'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['optimization']['gpu'] = -1
        config['object1']['image_filter'] = filt
        torch.manual_seed(0)
        d = torchmfbd.Deconvolution(config)
        d.add_frames(frames, sigma=0.005 * torch.ones(4, 5), id_object=0, id_diversity=0, diversity=0.0)
        d.deconvolve(infer_object=False, optimizer='lbfgs', simultaneous_sequences=200, n_iterations=3)
        out[filt] = d
    assert torch.equal(out['scharmer'].modes, out['scharmer_binary'].modes)
    assert not torch.equal(out['scharmer'].obj[0], out['scharmer_binary'].obj[0])
    assert torch.all(torch.isfinite(out['scharmer'].obj[0]))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_scharmer_filter_torch_matches_numpy(device):
    # The torch steps (used when the sums are on the GPU) give the gain of the numpy ones to the bit, with the noise
    # per frame as a fit passes it; run on the CPU as well, so that the path is checked without a GPU
    Q, P, r = _sums(n=32, n_seq=3, seed=2)
    sig = torch.tensor(np.random.default_rng(3).uniform(0.8, 1.2, (3, 5)))
    dummy = types.SimpleNamespace(rho=[torch.tensor(r)], _box_mean=torchmfbd.Deconvolution._box_mean,
                                  _box_mean_torch=torchmfbd.Deconvolution._box_mean_torch,
                                  _region_from_centre=torchmfbd.Deconvolution._region_from_centre)
    dummy._scharmer_filter_torch = lambda *a: torchmfbd.Deconvolution._scharmer_filter_torch(dummy, *a)
    ref = torchmfbd.Deconvolution.scharmer_filter(dummy, Q, P, sig, 0)
    out = torchmfbd.Deconvolution._scharmer_filter_torch(dummy, Q.to(device), P.to(device), sig.to(device), 0)
    assert out.device.type == device
    assert torch.equal(out.cpu(), ref)
    if device == "cuda":
        assert torch.equal(torchmfbd.Deconvolution.scharmer_filter(dummy, Q.cuda(), P.cuda(), sig.cuda(), 0).cpu(), ref)
