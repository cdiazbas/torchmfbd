import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchmfbd.az_average as az_average

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
    inside = np.where(rarr < cent_obs)

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

def apodize(frames, window):
    """
    Apodizes the input frames by subtracting the mean value and applying a window function.
    The mean value is computed along the last two dimensions of the input tensor.
    The window function is applied differently depending on the number of dimensions of the input tensor.
    The mean value is added back to the frames after applying the window function.
    
    Args:    
        frames (torch.Tensor): The input tensor containing the frames to be apodized. The tensor can have 2, 3, 4, or 5 dimensions.
    
    Returns:
        torch.Tensor: The apodized frames with the same shape as the input tensor.
    
    """
    
    if frames.device != window.device:
        frames = frames.to(window.device)

    ndim = frames.ndim

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

    frames_apodized += mean_val

    return frames_apodized



def azimuthal_power(self, image):        
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