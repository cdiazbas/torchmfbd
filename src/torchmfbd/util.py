from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchmfbd.az_average as az_average
import scipy.stats as stats

__all__ = ['aperture', 'psf_scale', 'apodize', 'azimuthal_power']

def aperture(npix=256, cent_obs=0.0, spider=0, overfill=1.0):
    """
    Compute the aperture image of a telescope
  
    Args:
        npix (int, optional): number of pixels of the aperture image
        cent_obs (float, optional): central obscuration fraction
        spider (int, optional): spider size in pixels
    
    Returns:
        real: returns the aperture of the telescope
    """
    illum = np.ones((npix,npix),dtype='d')
    x = np.arange(-npix/2,npix/2,dtype='d')
    y = np.arange(-npix/2,npix/2,dtype='d')

    xarr = np.outer(np.ones(npix,dtype='d'),x)
    yarr = np.outer(y,np.ones(npix,dtype='d'))

    rarr = np.sqrt(np.power(xarr,2) + np.power(yarr,2))/(npix/2)
    outside = np.where(rarr > 1.0/overfill)
    inside = np.where(rarr < cent_obs * 1.0/overfill)
    
    illum[outside] = 0.0
    if np.any(inside[0]):
        illum[inside] = 0.0

    if (spider > 0):        
        start = int(npix/2 - int(spider)/2)
        illum[start:start+int(spider),:] = 0.0
        illum[:,start:start+int(spider)] = 0.0

    return illum

def psf_scale(wavelength, telescope_diameter, simulation_pixel_size):
        """
        Return the PSF scale appropriate for the required pixel size, wavelength and telescope diameter
        The aperture is padded by this amount; resultant pix scale is lambda/D/psf_scale, so for instance full frame 256 pix
        for 3.5 m at 532 nm is 256*5.32e-7/3.5/3 = 2.67 arcsec for psf_scale = 3

        https://www.strollswithmydog.com/wavefront-to-psf-to-mtf-physical-units/#iv
                
        """
        return 206265.0 * wavelength * 1e-8 / (telescope_diameter * simulation_pixel_size)


class FiniteDifference(torch.nn.Module):
    def __init__(self):
        super(FiniteDifference, self).__init__()
        
        kernel = torch.zeros((2, 1, 2, 2))

        kernel[0, :, 0, 0] = 1.0
        kernel[0, :, 1, 0] = -1.0

        kernel[1, :, 0, 0] = 1.0
        kernel[1, :, 0, 1] = -1.0

        self.kernel = nn.Parameter(kernel)
        
    def forward(self, im):
        b, c, h, w = im.shape        
        return F.conv2d(im.reshape(b*c, 1, h, w), self.kernel)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def apodize(frames, window, gradient=False):
    """
    Apodizes the input frames by subtracting the mean value and applying a window function.
    The mean value is computed along the last two dimensions of the input tensor.
    The window function is applied differently depending on the number of dimensions of the input tensor.
    The mean value is added back to the frames after applying the window function.
    
    Args:    
        frames (torch.Tensor): The input tensor containing the frames to be apodized. The tensor can have 2, 3, 4, or 5 dimensions.
        window (torch.Tensor): The window function to be applied to the frames. The shape of the window should match the last two dimensions of the input tensor.
        gradient (bool, optional): If True, the global gradient of the image is removed. Default is False.
    
    Returns:
        torch.Tensor: The apodized frames with the same shape as the input tensor.
    
    """
    
    if frames.device != window.device:
        frames = frames.to(window.device)

    ndim = frames.ndim

    # If we want to remove the global gradient of the image
    if gradient:
        width, height = frames.shape[-2], frames.shape[-1]
        x = torch.linspace(0, 1, width)
        y = torch.linspace(0, 1, height)
        xv, yv = torch.meshgrid(x, y, indexing='ij')        
        xv_flat = xv.unsqueeze(1).reshape(-1, 1)
        yv_flat = yv.unsqueeze(1).reshape(-1, 1)
        coords = torch.cat((xv_flat, yv_flat), dim=1)
        A = torch.cat((coords, torch.ones(coords.shape[0], 1)), dim=1).to(frames.device)

        A_inv = torch.linalg.pinv(A)

        xv = xv.to(frames.device)
        yv = yv.to(frames.device)
        
        if ndim == 2:
            coeff = torch.einsum("ij,j->i", A_inv, frames.reshape(width * height))
            plane = coeff[0] * xv + coeff[1] * yv + coeff[2]
            
        elif ndim == 3:
            coeff = torch.einsum("ij,aj->ai", A_inv, frames.reshape(frames.shape[0], width * height))
            plane = coeff[:, 0][:, None, None] * xv[None, :, :] + coeff[:, 1][:, None, None] * yv[None, :, :] + coeff[:, 2][:, None, None]
            
        elif ndim == 4:
            avg_frames = torch.mean(frames, dim=1, keepdim=True)            
            coeff = torch.einsum("ij,abj->abi", A_inv, avg_frames.reshape(avg_frames.shape[0], avg_frames.shape[1], width * height))
            plane = coeff[:, :, 0][:, :, None, None] * xv[None, None, :, :] + coeff[:, :, 1][:, :, None, None] * yv[None, None, :, :] + coeff[:, :, 2][:, :, None, None]
            
        elif ndim == 5:
            coeff = torch.einsum("ij,abcj->abci", A_inv, frames.reshape(frames.shape[0], frames.shape[1], frames.shape[2], width * height))
            plane = coeff[:, :, :, 0][:, :, :, None, None] * xv[None, None, None, :, :] + coeff[:, :, :, 1][:, :, :, None, None] * yv[None, None, None, :, :] + coeff[:, :, :, 2][:, :, :, None, None]
            
        frames_apodized = frames - plane

    else:
        mean_val = torch.mean(frames, dim=(-1, -2), keepdim=True)
        frames_apodized = frames - mean_val
    
    if ndim == 2:
        frames_apodized *= window

    if ndim == 3:
        frames_apodized *= window[None, :, :]
    
    if ndim == 4:
        frames_apodized *= window[None, None, :, :]

    if ndim == 5:
        frames_apodized *= window[None, None, None, :, :]

    # Add the mean value back to the frames if only the mean was subtracted
    # If not, the gradient will be added at the end of the reconstruction
    if not gradient:        
        frames_apodized += mean_val
        plane = None
    
    return frames_apodized, plane    
    

def remove_fixed_pattern(frames, cutoff, coherence=0.5, power=2.0, rho_min=0.75, chunk=8):
    """
    Removes a pattern fixed on the detector from bursts of frames. Such a pattern is added after the telescope,
    so the frames do not blur it, and multi-frame deconvolution can only explain it as object structure divided
    by the OTF: near the diffraction cutoff it comes out as strong stripes. It is found as the frequencies where
    the frames of a burst agree with each other: solar structure near the cutoff changes with the seeing from
    frame to frame (coherence ~0.06), a pattern on the detector does not.

    For each burst, the frames are mirror-extended (so that the jump between opposite edges does not ring) and
    Fourier transformed. A frequency is taken as pattern where its coherence across frames,
    |mean_j F_j|^2 / mean_j |F_j|^2, exceeds `coherence`, its power exceeds `power` times the median of its ring,
    and it lies above `rho_min` times the diffraction cutoff (below that, solar structure is coherent too). The
    two thresholds are linear ramps (coherence +-0.1 and power +-0.5 around them) rather than steps, so the
    result is a continuous function of the frames. The mean over frames at those frequencies, so weighted, is the
    pattern, subtracted from every frame. The noise is left as it is, also beyond the cutoff, where the noise is
    estimated.

    The frames must not have been registered or destretched: the pattern then moves from frame to frame and is
    not found. With fewer than about 12 frames the coherence of noise alone becomes too high (measured with 6:
    285 false frequencies, 7 % of the solar power near the cutoff removed).

    Args:
        frames (torch.Tensor): bursts, shape (n_seq, n_frames, nx, ny), in detector coordinates.
        cutoff (float): the diffraction cutoff in cycles per pixel, pixel size / (lambda / D).
        coherence (float, optional): coherence threshold. Default 0.5.
        power (float, optional): power threshold, in units of the median power of the ring. Default 2.
        rho_min (float, optional): lowest frequency considered, in units of the cutoff. Default 0.75.
        chunk (int, optional): number of bursts transformed at a time. Default 8.

    Returns:
        torch.Tensor: the frames with the pattern removed, same shape and dtype.
        torch.Tensor: the pattern of each burst, shape (n_seq, nx, ny).
        torch.Tensor: the summed weight of the frequencies taken as pattern inside the diffraction limit in each
        burst (the number of frequencies, counting partial ones), shape (n_seq,).
    """

    n_seq, n_f, nx, ny = frames.shape
    mx, my = 2 * nx, 2 * ny

    kx = torch.fft.fftfreq(mx, dtype=torch.float64)
    ky = torch.fft.fftfreq(my, dtype=torch.float64)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    k = torch.sqrt(KX**2 + KY**2)
    rho = (k / cutoff).to(frames.device)
    ring = torch.round(k * max(mx, my)).long().to(frames.device)
    rings = [ring == r for r in range(int(ring.max()) + 1)]

    def mirror(x):
        top = torch.cat([x, torch.flip(x, dims=[-1])], dim=-1)
        return torch.cat([top, torch.flip(top, dims=[-2])], dim=-2)

    pattern = torch.zeros(n_seq, nx, ny, dtype=frames.dtype, device=frames.device)
    n_removed = torch.zeros(n_seq, dtype=torch.float64)

    def one_chunk(i0):
        x = frames[i0:i0 + chunk].to(torch.float64)
        F = torch.fft.fft2(mirror(x - torch.mean(x, dim=(-1, -2), keepdim=True)))
        P = torch.mean(torch.abs(F)**2, dim=1)
        C = torch.abs(torch.mean(F, dim=1))**2 / torch.clamp(P, min=1e-300)
        med = torch.zeros_like(P)
        for m in rings:
            med[:, m] = torch.quantile(P[:, m], 0.5, dim=-1)[:, None]
        w = torch.clamp((C - (coherence - 0.1)) / 0.2, min=0.0, max=1.0) * \
            torch.clamp((P / torch.clamp(med, min=1e-300) - (power - 0.5)) / 1.0, min=0.0, max=1.0) * (rho > rho_min)
        pat = torch.fft.ifft2(torch.fft.fft2(mirror(torch.mean(x, dim=1))) * w).real[:, :nx, :ny]
        pattern[i0:i0 + chunk] = pat.to(frames.dtype)
        n_removed[i0:i0 + chunk] = torch.sum(w * (rho <= 1.0), dim=(-1, -2)).cpu()

    # The chunks are independent: on the CPU they run on a thread pool, each the same computation, so the result is
    # the same bit for bit (one window of the time example, 25 bursts of 60 frames: 2.0 -> 0.8 s with 4 threads)
    starts = list(range(0, n_seq, chunk))
    workers = min(len(starts), torch.get_num_threads())
    if frames.device.type == 'cpu' and workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one_chunk, starts))
    else:
        for i0 in starts:
            one_chunk(i0)

    return frames - pattern[:, None, :, :], pattern, n_removed

def azimuthal_power_old(self, image):        
    """
    Compute the azimuthal power spectrum of an image.
    Args:
        image (numpy.ndarray): The input image for which the azimuthal power spectrum is to be computed.
    Returns:
        (f, p) (tuple): The normalized frequency array (f) and the azimuthally averaged power spectrum normalized by its first element (p).
    """
    _, freq_az = az_average.pspec(np.fft.fftshift(self.rho), azbins=1, binsize=1)
    k, power = az_average.power_spectrum(image)
    return 1.0/(freq_az * self.cutoff), power

def azimuthal_power_old2(image, d=1):

    n = image.shape[0] // 2
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift)**2

    freq = np.fft.fftfreq(image.shape[0], d=d)
        
    k, pow1d = az_average.azimuthalAverage(power, returnradii=True)

    ind = np.where(k < n)
    k = k[ind]
    pow1d = pow1d[ind]

    k = np.interp(k, np.arange(n), freq[0:n])
        
    return k, pow1d

def azimuthal_power(image, d=1, apodization=None, angles=None, range_angles=5):
    """
    Compute the azimuthal power spectrum of an image.
    Args:
        image (numpy.ndarray): The input image for which the azimuthal power spectrum is to be computed.
        d (float, optional): The pixel size in the image. Default is 1.
        apodization (int, optional): The size of the apodization window. Default is None.
        angles (list, optional): A list of angles in degrees for which to compute the azimuthal power spectrum. Default is None.
        range_angles (float, optional): The range of angles around each specified angle (+-range) to include in the computation. Default is 5.
    Returns:
        (kvals, Abins) (tuple): The normalized frequency array (kvals) and the azimuthally averaged power spectrum (Abins).
    """

    npix = image.shape[0]

    if apodization is not None:
        win = np.hanning(2*apodization)
        winOut = np.ones(npix)
        winOut[0:apodization] = win[0:apodization]
        winOut[-apodization:] = win[-apodization:]
        window = np.outer(winOut, winOut)
        mn = np.mean(image)
        image -= mn
        image *= window
        image += mn
        
    fourier_image = np.fft.fft2(image, norm="ortho")
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    if angles is not None:
        kangles = np.arctan2(kfreq2D[1], kfreq2D[0]).flatten()
        kangles = np.degrees(kangles)

        ind_angles = []

        for ang in angles:
            ind = np.where(np.abs(kangles - ang) < range_angles)
            ind_angles.append(ind[0])

        ind_angles = np.concatenate(ind_angles)

        knrm = knrm[ind_angles]
        fourier_amplitudes = fourier_amplitudes[ind_angles]
    
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])    
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)

    return kvals / npix, Abins

def num_modes_height(z, n_modes_0, r00, r0z, D, afov):
    """
    Calculate the number of modes as a function of height.

    This function computes the ratio of the number of modes at a given height 
    based on the Fried parameter (r0) and the aperture diameter (D), taking 
    into account the angular field of view (afov).

    Parameters:
        z (array-like): The height values at which the number of modes is to be calculated in km.
        n_modes_0 (float): The number of modes at the reference height.
        r00 (float): The Fried parameter at the reference height in cm.
        r0z (array-like): The Fried parameter values at different heights in cm.
        D (float): The aperture diameter in cm.
        afov (float): The angular field of view in arcsec

    Returns:
        array-like: The ratio of the number of modes at different heights 
        relative to the reference height.
    """
    afov = afov / 206265.0    
    z *= 1e5
    ratio = (r00 / r0z)**2 * (1.0 + (z * np.sin(afov)) / D)**2
    return ratio * n_modes_0

def orthogonalize(basis, pupil):
    """
    Orthogonalize the basis vectors using the Gram-Schmidt process via the QR decomposition.
    This is specially useful for the case of a pupil function that is not circular, 
    with strong obscurations

    Args:
        basis (torch.Tensor): The basis vectors to be orthogonalized.
        pupil (torch.Tensor): The pupil function used for orthogonalization.

    Returns:
        torch.Tensor: The orthogonalized basis vectors.
    """
    
    M, N, _ = basis.shape

    # Take into account the pupil function
    fun = basis * pupil[None, :, :]
    A = fun.reshape(M, N*N).T
    Q, R = np.linalg.qr(A)

    Q = Q.T.reshape(M, N, N)

    return Q

if __name__ == "__main__":
    # x = np.random.rand(256, 256)
    # k, pp = azimuthal_power(x, d=1, angles=[45, -45, 135, -135], range_angles=45)
    # k2, pp2 = azimuthal_power(x, d=1)

    # import matplotlib.pyplot as plt
    # plt.loglog(k, pp, label='Azimuthal Power Spectrum')
    # plt.loglog(k2, pp2, label='Azimuthal Power Spectrum (all angles)')

    z = np.array([0.0, 10.0, 20.0])  # Heights in km
    n_modes_0 = 44.0
    r00 = 5.0  # Fried parameter at reference height in cm
    r0z = np.array([5.0, 10.0, 15.0])  # Fried parameters at different heights in cm
    D = 100.0  # Aperture diameter in cm
    afov = 512 * 0.059  # Angular field of view in arcsec
    n_modes = num_modes_height(z, n_modes_0, r00, r0z, D, afov)

    print(n_modes)
    print((afov/3.0)**2/np.sum(n_modes/n_modes_0))

    z = np.array([0.0, 20.0])  # Heights in km
    n_modes_0 = 44.0
    r00 = 10.0  # Fried parameter at reference height in cm
    r0z = np.array([10.0, 35.0])  # Fried parameters at different heights in cm
    D = 100.0  # Aperture diameter in cm
    afov = 2.0 * 60. #256 * 0.059  # Angular field of view in arcsec
    n_modes = num_modes_height(z, n_modes_0, r00, r0z, D, afov)

    print(n_modes)

    print((afov/3.0)**2/np.sum(n_modes/n_modes_0))