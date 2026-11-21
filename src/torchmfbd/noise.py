from torchmfbd.noise_svd import noise_estimation
import torch

def compute_noise(frames):
    """
    Compute the noise characteristics of the images.
    Parameters:
    -----------
    frames : numpy.ndarray
        A 5D array of shape (n_scans, n_frames, nx, ny) representing the image frames.
    Returns:
    --------
    sigma : numpy.ndarray
        A 4D array of shape (n_obj, n_scans, n_frames, nx*ny) representing the estimated noise characteristics.
    """
    
    n_scans, n_frames, nx, ny = frames.shape
    
    sigma = noise_estimation(frames.numpy())

    sigma = torch.tensor(sigma.astype('float32')).to(frames.device)
    
    return sigma

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
