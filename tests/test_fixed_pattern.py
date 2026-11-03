import torchmfbd
import torch
import yaml
import warnings
from pathlib import Path
from torchmfbd import util

# CRISP-like sampling: 0.059 arcsec pixels, 1 m telescope at 8542 A
CUTOFF = 0.059 / (8542e-8 / 100.0 * 206265.0)

def _burst(n_frames, seed=0, n=64, sigma=0.02, pattern=True):
    """
    A k^-3 object moving by whole pixels from frame to frame, blurred by the diffraction OTF times a seeing
    attenuation exp(-a rho^(5/3)) that changes from frame to frame (a in [1, 4]), white noise, and optionally a
    pattern fixed on the detector: column offsets that differ above and below a seam, and a 2.3-pixel sinusoid
    beyond the cutoff.
    Returns the frames (1, n_frames, n, n), the clean frames and the pattern.
    """
    gen = torch.Generator().manual_seed(seed)
    k = torch.fft.fftfreq(n, dtype=torch.float64) * n
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    kk = torch.sqrt(kx**2 + ky**2).clamp(min=1.0)
    re, im = torch.randn(n, n, generator=gen, dtype=torch.float64), torch.randn(n, n, generator=gen, dtype=torch.float64)
    obj = torch.fft.ifft2((re + 1j * im) / kk**1.5).real
    obj = 1.0 + 0.15 * obj / obj.std()

    rho = torch.sqrt((kx / n)**2 + (ky / n)**2) / CUTOFF
    r = rho.clamp(max=1.0)
    otf_diff = 2.0 / torch.pi * (torch.arccos(r) - r * torch.sqrt(1.0 - r**2))

    clean = []
    for j in range(n_frames):
        shift = (0, 0) if j == 0 else tuple(torch.randint(-2, 3, (2,), generator=gen).tolist())
        a = 1.0 + 3.0 * torch.rand(1, generator=gen, dtype=torch.float64)
        img = torch.fft.ifft2(torch.fft.fft2(torch.roll(obj, shift, (0, 1))) * otf_diff * torch.exp(-a * rho**(5.0 / 3.0))).real
        clean.append(img + sigma * torch.randn(n, n, generator=gen, dtype=torch.float64))
    clean = torch.stack(clean)[None]

    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing='ij')
    col = torch.zeros(n, n, dtype=torch.float64)
    col[:n // 3] = 0.005 * torch.randn(n, generator=gen, dtype=torch.float64)[None]
    col[n // 3:] = 0.005 * torch.randn(n, generator=gen, dtype=torch.float64)[None]
    th = torch.tensor(2.5 * torch.pi / 180.0, dtype=torch.float64)
    sinus = 0.01 * torch.cos(2 * torch.pi * (torch.sin(th) * yy + torch.cos(th) * xx) / 2.3)
    pat = (col + sinus) if pattern else torch.zeros(n, n, dtype=torch.float64)

    return (clean + pat).float(), clean.float(), pat.float()

def _band(x, lo, hi):
    """Power of an image (Hann window) between lo and hi times the diffraction cutoff"""
    n = x.shape[-1]
    k = torch.fft.fftfreq(n, dtype=torch.float64)
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    rho = torch.sqrt(kx**2 + ky**2) / CUTOFF
    w = torch.hann_window(n, periodic=False, dtype=torch.float64)
    F = torch.fft.fft2((x.double() - x.double().mean()) * w[:, None] * w[None, :])
    return float(torch.sum(torch.abs(F[(rho > lo) & (rho <= hi)])**2))

def test_fixed_pattern_removed():

    frames, clean, pat = _burst(12)
    out, est, n_removed = util.remove_fixed_pattern(frames, CUTOFF)

    assert out.shape == frames.shape and out.dtype == frames.dtype, f'Expected {frames.shape} {frames.dtype}, got {out.shape} {out.dtype}'
    assert est.shape == (1, 64, 64), f'Expected (1, 64, 64), got {est.shape}'

    # The pattern near the cutoff, where the deconvolution amplifies it, and beyond the cutoff
    left_band = _band(pat - est[0], 0.75, 1.0) / _band(pat, 0.75, 1.0)
    left_beyond = _band(pat - est[0], 1.0, 2.0) / _band(pat, 1.0, 2.0)
    # At least half of it: how much is found depends on how much solar power there is near the cutoff
    assert left_band < 0.5, f'{left_band:.3f} of the pattern power left near the cutoff'
    assert left_beyond < 0.1, f'{left_beyond:.3f} of the pattern power left beyond the cutoff'

def test_fixed_pattern_clean():

    # Without a pattern: the noise beyond the cutoff (where the library estimates it) and the object are left alone
    frames, clean, pat = _burst(12, pattern=False)
    out, est, n_removed = util.remove_fixed_pattern(frames, CUTOFF)

    ratio = sum(_band(f, 1.0, 2.0) for f in out[0]) / sum(_band(f, 1.0, 2.0) for f in frames[0])
    assert abs(ratio - 1.0) < 0.01, f'Noise power beyond the cutoff changed by a factor {ratio:.4f}'
    assert float(est.std()) < 0.05 * 0.02, f'Pattern of rms {float(est.std()):.2e} found in frames without one (noise 0.02)'

def test_fixed_pattern_continuous():

    # Scaling the pattern smoothly changes the estimate smoothly: with step thresholds, frequencies switch on as
    # they cross a threshold and the estimate jumps (max/median step 1.5-1.9 over seeds 0-3, against 1.2-1.3 here)
    frames, clean, pat = _burst(12)
    alphas = torch.linspace(0.5, 1.5, 41)
    est = torch.stack([util.remove_fixed_pattern(clean + a * pat, CUTOFF)[1][0] for a in alphas])
    steps = torch.sqrt(torch.mean((est[1:] - est[:-1])**2, dim=(-1, -2)))
    ratio = float(steps.max() / steps.median())
    assert ratio < 1.4, f'Largest change between consecutive scalings is {ratio:.2f} times the median one'

def _config(**images):

    config_path = Path(__file__).parent / 'joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['images'].update({'n_pixel': 64, **images})
    config['optimization']['gpu'] = -1

    return config

def test_fixed_pattern_add_frames():

    frames, clean, pat = _burst(12)

    # On by default: the stored frames are the cleaned ones and the pattern is kept
    dec = torchmfbd.Deconvolution(_config())
    dec.add_frames(frames, sigma=0.02)
    assert dec.fixed_pattern[0] is not None, 'No pattern stored'
    expected, _, _ = util.remove_fixed_pattern(frames, CUTOFF)
    assert torch.allclose(dec.frames[0], expected), 'Stored frames are not the cleaned frames'

    # Switched off: the frames are stored as given
    dec = torchmfbd.Deconvolution(_config(remove_fixed_pattern=False))
    dec.add_frames(frames, sigma=0.02)
    assert dec.fixed_pattern[0] is None and torch.equal(dec.frames[0], frames), 'Frames changed with remove_fixed_pattern false'

    # Too few frames: skipped, with a warning
    dec = torchmfbd.Deconvolution(_config())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        dec.add_frames(frames[:, :6], sigma=0.02)
    assert any('Fixed-pattern removal skipped' in str(x.message) for x in w), 'No warning with 6 frames'
    assert dec.fixed_pattern[0] is None and torch.equal(dec.frames[0], frames[:, :6]), 'Frames changed with 6 frames'
