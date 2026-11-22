"""Repair a time series that was restored without a time term, from the restored objects alone.

Old reductions often keep only the restored objects, one per time step, fitted step by step. A step whose seeing was much
worse than its neighbours' then comes out smoother: its PSF fit absorbed part of the blur. Without the raw frames that
step can only borrow from its neighbours. repair_series deconvolves the series of objects again with the time path, one
"frame" per step, and removes only the blur that differs between steps: every step is blurred back with the mean OTF of
its patch over the series, since a blur common to all steps cannot be told from the Sun. Steps are weighted by the SNR
rule with the power of each step relative to the median step (snr_source: power). Only the weak steps are replaced.

Measured on 49-step synthetic series with a known truth (2026-11-22): replacing only the steps with less than half the
median power lowered the mean object error in 6 of 6 series (-3 to -15 %; -3 and -8 % on the two series not used to set
the threshold); replacing every step raised it in one series by 12 %. The full method from the raw frames did about twice
as well: this repairs what an old product allows, it does not replace a new reduction.
"""
import copy
import numpy as np
import torch

__all__ = ['repair_series']


def _cutoff_rho(config, n):
    """|frequency| / diffraction cutoff on an n x n FFT grid, from the telescope, wavelength and pixel of the config."""
    lam_cm = float(config['object1']['wavelength']) * 1e-8
    cutoff = config['telescope']['diameter'] / lam_cm / 206265.0 * config['images']['pix_size']       # cycles per pixel
    f = np.fft.fftfreq(n)
    return torch.tensor(np.hypot(*np.meshgrid(f, f, indexing='ij')) / cutoff)


def repair_series(objects, config, weak_fraction=0.5, stride=None, n_iterations=10):
    """
    Repair the weak steps of a series restored without time term.

    Parameters
    ----------
    objects : array (n_t, n, n)
        The restored object of every time step (one object, one field).
    config : dict
        A torchmfbd configuration for the patches (images.n_pixel, telescope, object1.wavelength, psf, optimization); its
        time weight is used, the time path and the SNR rule are set here.
    weak_fraction : float
        A step is repaired when its power at 0.1-0.3 of the cutoff is below this fraction of the median step.
    stride : int or None
        Stride of the patches; None takes the one closest to 32 px that tiles the field exactly.
    n_iterations : int
        LBFGS iterations.

    Returns
    -------
    repaired : numpy.ndarray (n_t, m, m)
        The series on the unpatchified field (m <= n, centred), the weak steps replaced.
    weak : numpy.ndarray (n_t,) of bool
        The steps that were replaced.
    """
    import torchmfbd
    x = np.asarray(objects, dtype='float64')
    n_t, n = x.shape[0], x.shape[-1]
    npix = int(config['images']['n_pixel'])
    if stride is None:
        k = max(1, int(round((n - npix) / 32.0)))
        stride = max(1, (n - npix) // k) if n > npix else npix
    stride = min(stride, npix - 2 * 6)                       # the mosaic is unpatchified with an apodization of 6 px
    cfg = copy.deepcopy(config)
    cfg.pop('object2', None)
    cfg['optimization']['filter_loss'] = 'wiener_time'
    reg = cfg.setdefault('regularization', {}) or {}
    cfg['regularization'] = reg
    time_cfg = dict(reg.get('time', {}) or {})
    time_cfg.setdefault('lambda', 5e-4)
    time_cfg.update(snr_threshold=12.0, snr_reference='window', snr_source='power')
    reg['time'] = time_cfg

    # power of every step relative to the median step: the steps to repair
    rho_full = _cutoff_rho(cfg, n).numpy()
    band = (rho_full > 0.1) & (rho_full < 0.3)
    power = np.array([(np.abs(np.fft.fft2(s - s.mean()))**2)[band].mean() for s in x])
    weak = power / np.median(power) < weak_fraction

    patchify = torchmfbd.Patchify4D()
    fr = torch.tensor(x / x.mean(axis=(-1, -2), keepdims=True), dtype=torch.float32)
    pt = torch.stack([patchify.patchify(fr[t][None, None], patch_size=npix, stride_size=stride, flatten_sequences=True)[:, 0]
                      for t in range(n_t)], dim=1)                                                  # (L, n_t, npix, npix)
    L = pt.shape[0]

    # noise of a restored object: its power at 0.7-0.95 of the cutoff, where it is mostly amplified noise (an upper bound)
    rho = _cutoff_rho(cfg, npix)
    F = torch.fft.fft2(pt - pt.mean(dim=(-1, -2), keepdim=True))
    sigma = (torch.sqrt((torch.abs(F)**2)[..., (rho > 0.7) & (rho < 0.95)].mean(-1)) / npix)[:, :, None]

    dec = torchmfbd.Deconvolution(cfg)
    dec.add_frames(pt[:, :, None], sigma=sigma, id_object=0, id_diversity=0, diversity=0.0)
    if L > 1:
        dec.set_neighbours(patchify.neighbours())
    dec.deconvolve(infer_object=False, optimizer='lbfgs', simultaneous_sequences=L * n_t, n_iterations=n_iterations)

    # relative deconvolution: blur every step back with the mean OTF of its patch over the series
    obj = dec.obj[0].reshape(L, n_t, npix, npix).cpu().double()
    psf = torch.cat([b[0] for b in dec.psf_seq], 0).cpu().reshape(L, n_t, -1, npix, npix)[:, :, 0].double()
    otf = torch.fft.fft2(psf)
    obj = torch.fft.ifft2(torch.fft.fft2(obj) * otf.mean(dim=1, keepdim=True)).real
    out = np.stack([np.squeeze(patchify.unpatchify(obj[:, t][:, None].float(), apodization=6, weight_type='cosine', weight_params=30).numpy())
                    for t in range(n_t)]).astype('float64')
    m = out.shape[-1]; c0 = (n - m) // 2
    prod = x[:, c0:c0 + m, c0:c0 + m]
    out = out * (prod.mean(axis=(-1, -2)) / out.mean(axis=(-1, -2)))[:, None, None]
    repaired = np.where(weak[:, None, None], out, prod)
    return repaired, weak
