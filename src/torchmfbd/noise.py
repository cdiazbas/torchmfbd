import math
import torch

def compute_noise(frames):
    """
    Noise of each frame, from the singular values of the frame (Gavish & Donoho 2014, as noise_svd.noise_estimation).

    The singular values come from the eigenvalues of the frame's covariance, in float64 on the CPU with torch: about
    four times faster than the SVD in numpy it replaces (1.9 s against 7.9 s for 125 x 12 frames of 128 px,
    2026-11-21) and more precise, since the numpy path worked in float32 (the two differ by up to 2.6e-3). The result
    does not depend on the GPU, nor on which other frames are passed with a frame.

    Parameters:
    -----------
    frames : torch.Tensor
        (n_scans, n_frames, nx, ny) frames.

    Returns:
    --------
    sigma : torch.Tensor
        (n_scans, n_frames) noise standard deviation of each frame, float32, on the device of the frames.
    """
    m, n = frames.shape[-2:]
    beta = 1.0 * m / n
    w = (8.0 * beta) / (beta + 1 + math.sqrt(beta**2 + 14 * beta + 1))
    lambda_star = math.sqrt(2.0 * (beta + 1) + w)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43

    x = torch.as_tensor(frames).detach().to('cpu', torch.float64)
    cov = x.transpose(-2, -1) @ x if m > n else x @ x.transpose(-2, -1)
    singular = torch.sqrt(torch.linalg.eigvalsh(cov).clamp(min=0))     # ascending
    k = singular.shape[-1]
    median = singular[..., k // 2] if k % 2 else 0.5 * (singular[..., k // 2 - 1] + singular[..., k // 2])   # as np.median
    sigma = median / (math.sqrt(n) * lambda_star / omega)

    return sigma.to(torch.float32).to(frames.device)


def step_snr(frames, rho, band=(0.1, 0.3), noise_from=1.1, fourier=False):
    """
    Signal-to-noise ratio of each time step, measured from its own frames.

    The power of the frames, summed over the frames of the step, is averaged between band[0] and band[1] times the
    diffraction cutoff (the signal) and beyond noise_from times the cutoff, where the telescope passes nothing (the
    noise). The ratio (signal - noise) / noise says how far above the noise a step's data are where the seeing decides
    the restoration: below 1 the frames carry almost nothing of the object there.

    Parameters
    ----------
    frames : torch.Tensor
        (..., n_frames, nx, ny) images, or their 2-D FFTs with fourier=True. Images have their mean removed and a
        Hann window applied before the transform.
    rho : torch.Tensor
        (nx, ny) modulus of the frequency in units of the diffraction cutoff, in the FFT layout (Deconvolution.rho[i]).
    band : tuple of float
        Frequencies of the signal, in units of the cutoff.
    noise_from : float
        Frequencies above this (in units of the cutoff) measure the noise.
    fourier : bool
        True if frames are already Fourier transformed (for instance apodized patches).

    Returns
    -------
    torch.Tensor
        (...) the ratio per step, at least 1e-6.
    """
    rho = torch.as_tensor(rho, device=frames.device)
    noise = rho > noise_from
    signal = (rho > band[0]) & (rho < band[1])
    if not torch.any(noise):
        raise ValueError(f"step_snr needs frequencies beyond {noise_from} times the diffraction cutoff to measure the noise; "
                         f"the largest in these images is {float(rho.max()):.2f} times the cutoff")
    if not fourier:
        nx, ny = frames.shape[-2:]
        win = torch.outer(torch.hann_window(nx, periodic=False, dtype=frames.dtype, device=frames.device),
                          torch.hann_window(ny, periodic=False, dtype=frames.dtype, device=frames.device))
        frames = torch.fft.fft2((frames - frames.mean(dim=(-1, -2), keepdim=True)) * win)
    power = torch.sum(torch.abs(frames)**2, dim=-3)
    p_noise = power[..., noise].mean(-1)
    p_signal = power[..., signal].mean(-1)
    return ((p_signal - p_noise) / p_noise).clamp(min=1e-6)
