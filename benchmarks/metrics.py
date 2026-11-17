"""Error measures against a known truth. Every number the benchmark reports comes from here, and every function checks
its own preconditions: a number that cannot be trusted raises instead of being returned.

Units. The library is given frames divided by their own mean, so the object comes out in those units. The truth has to
be in the same units (`truth_units`); `object_error` refuses an object and a truth whose means differ by more than
RATIO_TOL. Without this check, a truth in other units (whole-image mean 0.76-0.85, or not divided at all: 0.91-1.05)
silently adds a contrast error of up to 30 %.

PSF reference. A periodic PSF model of m pixels can only represent the true PSF folded onto an m-px period
(`fold_psf`), whose OTF is the full OTF sampled on the m-px grid. Cropping the true PSF to m px and renormalising differs
from it by 2.3 / 3.4 / 9.8 % (relative |OTF|, median) at D/r0 3 / 5 / 10 for m = 64.

Which PSF measure. `transfer_error` (the transfer function the object actually sees) predicts the object error; the
per-frame `psf_error` does not (Spearman +0.71 / +0.88 on real / synthetic scenes against -0.18 / -0.47).

Control (run.py). Per scene, the object made from the true PSFs may not be worse than the code's by more than
CONTROL_SCENE_TOL; over each set, its mean error must be lower (`check_control`). If either fails, the metric or the
truth is wrong. The true-PSF object is a reference, not an optimum (same filter and apodization): with large patches a
good fit gets within a few % of it and, in single scenes, 1-2 % below it. Units errors are caught by `object_error`.
"""
import numpy as np
from skimage.registration import phase_cross_correlation

PIX, LD = 0.059, 0.17619          # arcsec per pixel; lambda / D in arcsec (8542 A, 1 m): the cutoff is PIX/LD cycles/px
MEAN_TOL = 0.005
RATIO_TOL = 0.03                  # a legitimate 1.9 % was seen without apodization; the units errors found were 5-30 %
BANDS = [(0.05, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.95)]   # fractions of the diffraction cutoff


def truth_units(truth):
    """The truth in the units of frames divided by their own mean."""
    t = truth / truth.mean()
    assert abs(t.mean() - 1) < MEAN_TOL
    return t


def _align(o, tr):
    """Shift o onto tr (sub-pixel phase correlation, upsampled x20, Fourier shift)."""
    sh, _, _ = phase_cross_correlation(tr - tr.mean(), o - o.mean(), upsample_factor=20, normalization=None)
    fy = np.fft.fftfreq(o.shape[0])[:, None]; fx = np.fft.fftfreq(o.shape[1])[None]
    return np.fft.ifft2(np.fft.fft2(o) * np.exp(-2j * np.pi * (fy * sh[0] + fx * sh[1]))).real


def _check_units(o, tr):
    # o and tr went through the same patch filtering and recombination, which shift both means alike (a few %): the
    # deconvolution keeps the mean, a units error does not
    if abs(o.mean() / tr.mean() - 1) > RATIO_TOL:
        raise ValueError(f"object mean {o.mean():.4f} against truth mean {tr.mean():.4f}: units differ (truth_units?)")


def object_error(o, tr, b=16):
    """rms(o - tr) / std(tr) after sub-pixel alignment, means removed, a border of b px dropped."""
    _check_units(o, tr)
    o = _align(o, tr)
    x, y = (o - o.mean())[b:-b, b:-b], (tr - tr.mean())[b:-b, b:-b]
    return float(np.sqrt(np.mean((x - y)**2)) / y.std())


def object_error_bestscale(o, tr, b=16):
    """As object_error with the object scaled by its least-squares factor: the part of the error that is not contrast."""
    o = _align(o, tr)
    x, y = (o - o.mean())[b:-b, b:-b], (tr - tr.mean())[b:-b, b:-b]
    c = np.sum(x * y) / np.sum(x * x)
    return float(np.sqrt(np.mean((c * x - y)**2)) / y.std())


def band_error(o, tr, b=16):
    """Relative object error per frequency band (BANDS, fractions of the cutoff): sqrt(sum |O - T|^2 / sum |T|^2).
    1 = nothing right in that band; above 1 = power that is not in the truth."""
    _check_units(o, tr)
    o = _align(o, tr)
    o = o[b:-b, b:-b] - o[b:-b, b:-b].mean(); t = tr[b:-b, b:-b] - tr[b:-b, b:-b].mean()
    n = o.shape[0]; w = np.outer(np.hanning(n), np.hanning(o.shape[1]))
    E, T = np.abs(np.fft.fft2((o - t) * w))**2, np.abs(np.fft.fft2(t * w))**2
    ky, kx = np.fft.fftfreq(n), np.fft.fftfreq(o.shape[1])
    rho = np.hypot(*np.meshgrid(ky, kx, indexing="ij")) / (PIX / LD)
    return [float(np.sqrt(E[(rho > a) & (rho <= c)].sum() / T[(rho > a) & (rho <= c)].sum())) for a, c in BANDS]


def fold_psf(p, m):
    """PSF (origin at the corner) folded onto an m-px period, sum 1. If m does not divide the PSF size, the PSF is
    zero-padded to the next multiple of m first."""
    n = p.shape[0]
    if n % m:
        k = -(-n // m) * m; q = np.zeros((k, k)); h = n // 2
        q[:h, :h] = p[:h, :h]; q[:h, -(n - h):] = p[:h, h:]; q[-(n - h):, :h] = p[h:, :h]; q[-(n - h):, -(n - h):] = p[h:, h:]
        p, n = q, k
    w = p.reshape(n // m, m, n // m, m).sum(axis=(0, 2))
    return w / w.sum()


def psf_band(m):
    """Frequencies 0 < rho <= 0.95 of the cutoff on an m-px grid."""
    k = np.fft.fftfreq(m) * m
    q = np.hypot(*np.meshgrid(k, k, indexing="ij")) / (m * PIX / LD)
    return (q > 0) & (q <= 0.95)


def psf_error(pe, true_psfs):
    """Median over patches and frames of || |OTF_est| - |OTF_true| || / || |OTF_true| || inside 0.95 of the cutoff.
    pe: (patches, frames, m, m) fitted PSFs; true_psfs: (frames, N, N), or (patches, frames, N, N) when they depend on
    the position in the field."""
    m = pe.shape[-1]; band = psf_band(m)
    tp = true_psfs if true_psfs.ndim == 4 else np.broadcast_to(true_psfs, (pe.shape[0],) + true_psfs.shape)
    e = []
    for i in range(pe.shape[0]):
        for j in range(pe.shape[1]):
            ref = np.abs(np.fft.fft2(fold_psf(tp[i, j], m)))
            e.append(np.linalg.norm((np.abs(np.fft.fft2(pe[i, j])) / pe[i, j].sum() - ref)[band]) / np.linalg.norm(ref[band]))
    return float(np.median(e))


def transfer_error(pe, otf_true, truth_patches):
    """Error of the transfer function the object sees, R(k) = sum_j conj(T_est_j) T_true_j / sum_j |T_est_j|^2 (R = 1
    for perfect PSFs), as rms |R - 1| weighted by the true object power inside 0.95 of the cutoff; median over patches.
    pe: (patches, frames, m, m); otf_true: (frames, m, m) or (patches, frames, m, m), with the frames' shifts, DC = 1;
    truth_patches: (patches, m, m)."""
    band = psf_band(pe.shape[-1])
    te = np.fft.fft2(pe / pe.sum(axis=(-1, -2), keepdims=True))
    ot = otf_true if otf_true.ndim == 4 else otf_true[None]
    R = np.sum(np.conj(te) * ot, axis=1) / np.maximum(np.sum(np.abs(te)**2, axis=1), 1e-12)
    w = np.abs(np.fft.fft2(truth_patches - truth_patches.mean(axis=(-1, -2), keepdims=True)))**2
    e = [np.sqrt(np.sum(w[i][band] * np.abs(R[i][band] - 1)**2) / np.sum(w[i][band])) for i in range(pe.shape[0])]
    return float(np.median(e))


CONTROL_SCENE_TOL = 1.05


def check_control(err_true_psf, err_code, label="", margin=1.0):
    """The true-PSF error must be below margin times the code's (run.py: margin 1 on the means over a set)."""
    if not err_true_psf < margin * err_code:
        raise RuntimeError(f"control failed {label}: the object from the true PSFs ({err_true_psf:.4f}) is not "
                           f"better than the code's ({err_code:.4f}); check the truth units and the PSF reference")
