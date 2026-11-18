"""Bursts with a known truth, reproducible from their seed. Telescope and sampling of the CRISP 8542 A example: 1 m
aperture, 0.059"/px, so the diffraction cutoff sits at 0.67 of the Nyquist frequency.

Every scene is a dict: name, frames (J, n, n), truth (n, n, in the units of frames divided by their mean), shifts (the
integer image motion of each frame, frame 0 fixed), and either psfs (J, 128, 128), the same PSFs over the whole field,
or psf_at(j, y, x), the PSF of frame j at a position of the field (anisoplanatic scenes). PSFs have their origin at the
corner and sum 1.

Sets:
  synth        13 synthetic bursts, 128 px: k^-3 object of contrast 0.15, Kolmogorov D/r0 5 (tip-tilt removed),
               image motion rms 1.5 px, white noise. Needs nothing.
  aniso-synth  12 synthetic 256-px scenes with two-layer seeing (half the phase variance at the pupil, half in a layer
               at `alt` m that shifts with the field angle), D/r0 5 and 10. The image is sum_b W_b (O * PSF_b) over a
               64-px grid of PSFs with bilinear weights. Needs nothing.
  real         18 crops (256 px) of the sharpest frame of the example's burst, at D/r0 3, 5 and 10. Needs the burst file.
  aniso-real   12 such crops with the two-layer seeing of aniso-synth. Needs the burst file.
  aniso-time   6 synthetic 256-px scenes evolving over 4 time steps, each step with its own two-layer seeing, for the
               time path (filter_loss wiener_time). Needs nothing.
  real-time    as aniso-time, with the real solar evolution of the SST/CRISP 8542 time sequence: 3 windows of 4
               steps, D/r0 5 and 10. Needs the sequence file.
Hold-out (holdout=True, run.py --holdout): every seed shifted by HOLDOUT, names ending in _h. Synthetic objects, seeing,
motion and noise all change; the real sets keep their crops of the burst and change the seeing, motion and noise. Use
them to confirm a choice made on the default scenes.
The burst file is spot_20200727_083509_8542_npix512_original.h5, the data of examples/spot_8542, and the sequence file
sequence_8542_npix256_aligned.h5, the data of examples/spot_8542_time (neither is in the repository; see README.md for
where to get them).
"""
import numpy as np

N = 128                                   # pupil and PSF grid
PIX, LD = 0.059, 0.17619                  # arcsec per pixel; lambda / D in arcsec
DP = N * PIX / LD                         # pupil diameter in pixels of the N-px grid
_yy, _xx = np.meshgrid(np.arange(N) - N / 2, np.arange(N) - N / 2, indexing="ij")
PUPIL = (np.hypot(_yy, _xx) <= DP / 2).astype(float)
_PM = PUPIL > 0
HOLDOUT = 50000                           # added to every seed for the hold-out scenes (--holdout)
CROPS = [(64, 64), (64, 128), (128, 64), (128, 128), (64, 0), (0, 64)]      # corners of the 384-px crops of the burst


def _tilt_free(ph):
    A = np.stack([np.ones(_PM.sum()), _yy[_PM], _xx[_PM]], 1)
    out = ph.copy(); out[_PM] -= A @ np.linalg.lstsq(A, ph[_PM], rcond=None)[0]
    return out


def screen(rng, d_r0):
    """Kolmogorov phase over the pupil with variance 0.134 (D/r0)^(5/3) rad^2 after removing piston and tip-tilt."""
    n = 2 * N; f = np.fft.fftfreq(n); fr = np.hypot(*np.meshgrid(f, f, indexing="ij")); fr[0, 0] = np.inf
    ph = np.fft.ifft2((rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * fr**(-11 / 6)).real[:N, :N]
    ph = _tilt_free(ph)
    return ph * np.sqrt(0.134 * d_r0**(5 / 3) / ph[_PM].var())


def psf_from(phase):
    p = np.abs(np.fft.fft2(PUPIL * np.exp(1j * phase)))**2
    return p / p.sum()


def _object(rng, n):
    k = np.fft.fftfreq(n) * n; kk = np.hypot(*np.meshgrid(k, k, indexing="ij")); kk[0, 0] = 1
    o = np.fft.ifft2((rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / kk**1.5).real
    return 1 + 0.15 * o / o.std()


def _motion(rng, j):
    return (0, 0) if j == 0 else tuple(int(v) for v in np.rint(1.5 * rng.standard_normal(2)))


def synthetic(seed, n_frames=12, noise=0.02, d_r0=5.0, holdout=False):
    """One 128-px synthetic burst."""
    rng = np.random.default_rng(1000 + seed + (HOLDOUT if holdout else 0))
    obj = _object(rng, N)
    frames, psfs, shifts = [], [], []
    for j in range(n_frames):
        p = psf_from(screen(rng, d_r0))
        img = np.fft.ifft2(np.fft.fft2(obj) * np.fft.fft2(p)).real
        sh = _motion(rng, j)
        frames.append(np.roll(img, sh, (0, 1)) + noise * rng.standard_normal((N, N))); psfs.append(p); shifts.append(sh)
    frames = np.array(frames)
    return dict(name=f"synth_s{seed}" + ("_h" if holdout else ""), frames=frames, truth=obj / frames.mean(), psfs=np.array(psfs), shifts=shifts)


def _embed(p, nb):
    """128-px PSF (origin at the corner) on an nb-px grid, origin at the corner."""
    P = np.zeros((nb, nb)); P[nb // 2 - N // 2:nb // 2 + N // 2, nb // 2 - N // 2:nb // 2 + N // 2] = np.fft.fftshift(p)
    return np.fft.ifftshift(P)


def _degrade(gt384, rng, n_frames, noise, d_r0):
    """Isoplanatic degradation of a 384-px scene, cropped to its central 256 px (no wrap-around)."""
    frames, psfs, shifts = [], [], []
    F0 = np.fft.fft2(gt384)
    for j in range(n_frames):
        p = psf_from(screen(rng, d_r0))
        img = np.fft.ifft2(F0 * np.fft.fft2(_embed(p, 384))).real
        sh = _motion(rng, j)
        frames.append(np.roll(img, sh, (0, 1))[64:320, 64:320] + noise * rng.standard_normal((256, 256)))
        psfs.append(p); shifts.append(sh)
    return np.array(frames), np.array(psfs), shifts


def _degrade_aniso(gt384, rng, n_frames, noise, d_r0, alt):
    """Two-layer degradation of a 384-px scene: product convolution over a 6 x 6 grid of PSFs, cropped to 256 px."""
    NB, NS = 384, 1024
    alpha = alt * (PIX * np.pi / 180 / 3600) / (1.0 / DP)          # pupil px the upper layer moves per image px
    def big_screen(var):
        f = np.fft.fftfreq(NS); fr = np.hypot(*np.meshgrid(f, f, indexing="ij")); fr[0, 0] = np.inf
        ph = np.fft.ifft2((rng.standard_normal((NS, NS)) + 1j * rng.standard_normal((NS, NS))) * fr**(-11 / 6)).real
        c = _tilt_free(ph[NS // 2 - N // 2:NS // 2 + N // 2, NS // 2 - N // 2:NS // 2 + N // 2])
        return ph * np.sqrt(var / c[_PM].var())
    layers = [(screen(rng, d_r0 / 2**(3 / 5)), big_screen(0.5 * 0.134 * d_r0**(5 / 3))) for _ in range(n_frames)]
    def psf_384(j, py, px):
        g, hs = layers[j]
        oy = int(round(NS / 2 - N / 2 + alpha * (py - NB / 2))); ox = int(round(NS / 2 - N / 2 + alpha * (px - NB / 2)))
        return psf_from(g + hs[oy:oy + N, ox:ox + N])
    grid = np.arange(32, NB, 64)
    def hat(c):
        w = np.clip(1 - np.abs(np.arange(NB)[:, None] - c[None, :]) / 64.0, 0, None)
        w[:grid[0], 0] = 1; w[grid[-1]:, -1] = 1
        return w / w.sum(axis=1, keepdims=True)
    W = hat(grid); F0 = np.fft.fft2(gt384)
    frames, shifts = [], []
    for j in range(n_frames):
        img = np.zeros((NB, NB))
        for a, cy in enumerate(grid):
            for b, cx in enumerate(grid):
                img += np.outer(W[:, a], W[:, b]) * np.fft.ifft2(F0 * np.fft.fft2(_embed(psf_384(j, cy, cx), NB))).real
        sh = _motion(rng, j)
        frames.append(np.roll(img, sh, (0, 1))[64:320, 64:320] + noise * rng.standard_normal((256, 256))); shifts.append(sh)
    return np.array(frames), (lambda j, y, x: psf_384(j, y + 64, x + 64)), shifts


def time_scenes(n_frames=12, noise=0.02, alt=5000.0, n_steps=4, evolve=0.3, holdout=False):
    """Set aniso-time: 6 synthetic 256-px scenes that evolve in time, n_steps bursts each. The object of step t is
    cos(evolve t) (A - 1) + sin(evolve t) (B - 1) + 1 for two independent k^-3 fields A and B (the evolution of
    timestep/synth.py), and every step gets its own two-layer seeing (D/r0 5 and 10, upper layer at `alt`).
    Each scene: name, frames (T, J, 256, 256), truth (T, 256, 256), psf_at[t](j, y, x), shifts[t], all per step."""
    for d_r0 in (5.0, 10.0):
        for s in range(3):
            rng = np.random.default_rng(8000 + 100 * int(d_r0) + s + (HOLDOUT if holdout else 0))
            A, B = _object(rng, 384), _object(rng, 384)
            frames, truths, psf_at, shifts = [], [], [], []
            for t in range(n_steps):
                g384 = np.cos(evolve * t) * (A - 1) + np.sin(evolve * t) * (B - 1) + 1
                f, p, sh = _degrade_aniso(g384, rng, n_frames, noise, d_r0, alt)
                frames.append(f); psf_at.append(p); shifts.append(sh)
                truths.append(g384[64:320, 64:320] / f.mean())
            yield dict(name=f"time_r0{d_r0:g}_s{s}" + ("_h" if holdout else ""), frames=np.array(frames), truth=np.array(truths), psf_at=psf_at,
                       shifts=shifts, time=True)


def _sharp(x):
    gy, gx = np.gradient(x)
    return np.sqrt(np.mean(gx**2 + gy**2)) / x.mean()


def real_time_scenes(seq_path, n_frames=12, noise=0.02, alt=5000.0, n_steps=4, n_windows=3, holdout=False):
    """Set real-time: the solar evolution of the SST/CRISP 8542 time sequence (49 steps of 12 frames, 256 px). The truth
    of a step is its sharpest wide-band frame, which keeps its own residual seeing and noise. The windows are the
    n_windows non-overlapping runs of n_steps consecutive steps whose least sharp truth is sharpest, so that a change of
    the residual seeing between steps is not taken for solar evolution. Each step is extended to 384 px by reflection
    (the scene outside the field is its mirror image; the forward model stays exact) and gets its own two-layer seeing
    (D/r0 5 and 10, upper layer at `alt`). Scenes as in aniso-time."""
    import h5py
    with h5py.File(seq_path, "r") as f:
        im = f["im"][:, 0].astype("float64")
    gt = []
    for t in range(im.shape[0]):
        b = im[t, int(np.argmax([_sharp(x[32:-32, 32:-32]) for x in im[t]]))]
        gt.append(b / b.mean())
    s = np.array([_sharp(g[32:-32, 32:-32]) for g in gt])
    worst = np.array([s[t:t + n_steps].min() for t in range(len(gt) - n_steps + 1)])
    starts = []
    for t in np.argsort(-worst):
        if all(abs(int(t) - u) >= n_steps for u in starts):
            starts.append(int(t))
        if len(starts) == n_windows:
            break
    for d_r0 in (5.0, 10.0):
        for w, t0 in enumerate(sorted(starts)):
            rng = np.random.default_rng(9000 + 100 * int(d_r0) + w + (HOLDOUT if holdout else 0))
            frames, truths, psf_at, shifts = [], [], [], []
            for t in range(t0, t0 + n_steps):
                g384 = np.pad(gt[t], 64, mode="reflect")
                f, p, sh = _degrade_aniso(g384, rng, n_frames, noise, d_r0, alt)
                frames.append(f); psf_at.append(p); shifts.append(sh)
                truths.append(g384[64:320, 64:320] / f.mean())
            yield dict(name=f"rtime_r0{d_r0:g}_t{t0}" + ("_h" if holdout else ""), frames=np.array(frames), truth=np.array(truths),
                       psf_at=psf_at, shifts=shifts, time=True)


def _burst_truth(h5_path):
    import h5py
    with h5py.File(h5_path, "r") as f:
        im = f["im"][0, 0].astype("float64")
    def sharp(x):
        gy, gx = np.gradient(x); return np.sqrt(np.mean(gx**2 + gy**2)) / x.mean()
    gt = im[int(np.argmax([sharp(x[64:-64, 64:-64]) for x in im]))]
    return gt / gt.mean()


def scene_set(name, n_frames=12, noise=0.02, alt=5000.0, h5_path=None, holdout=False, seq_path=None):
    """The scenes of one set (see the module docstring), generated one at a time."""
    if name == "real-time":
        if seq_path is None:
            raise FileNotFoundError("the real-time scenes need the time sequence: pass --seq (see benchmarks/README.md)")
        yield from real_time_scenes(seq_path, n_frames, noise, alt, holdout=holdout)
        return
    if name == "synth":
        for seed in range(13):
            yield synthetic(seed, n_frames, noise, holdout=holdout)
        return
    if name == "aniso-time":
        yield from time_scenes(n_frames, noise, alt, holdout=holdout)
        return
    if name in ("real", "aniso-real"):
        if h5_path is None:
            raise FileNotFoundError("the real scenes need the burst file: pass --h5 (see benchmarks/README.md)")
        gt = _burst_truth(h5_path)
        sources = [(f"s{s}", gt[y0:y0 + 384, x0:x0 + 384]) for s, (y0, x0) in enumerate(CROPS)]
    elif name == "aniso-synth":
        sources = [(f"s{s}", _object(np.random.default_rng(7000 + s + (HOLDOUT if holdout else 0)), 384)) for s in range(6)]
    else:
        raise ValueError(f"unknown scene set {name}")
    tag = {"real": "real", "aniso-real": "aniso_real", "aniso-synth": "aniso_synth"}[name]
    seeing = (3.0, 5.0, 10.0) if name == "real" else (5.0, 10.0)
    for d_r0 in seeing:
        for s, (label, g384) in enumerate(sources):
            rng = np.random.default_rng(2000 + 100 * int(d_r0) + s + (HOLDOUT if holdout else 0))
            truth = g384[64:320, 64:320] / g384[64:320, 64:320].mean()
            sc = dict(name=f"{tag}_r0{d_r0:g}_{label}" + ("_h" if holdout else ""))
            if name == "real":
                frames, psfs, shifts = _degrade(g384, rng, n_frames, noise, d_r0)
                sc.update(psfs=psfs)
            else:
                frames, psf_at, shifts = _degrade_aniso(g384, rng, n_frames, noise, d_r0, alt)
                sc.update(psf_at=psf_at)
            sc.update(frames=frames, truth=truth * (g384[64:320, 64:320].mean() / frames.mean()), shifts=shifts)
            yield sc
