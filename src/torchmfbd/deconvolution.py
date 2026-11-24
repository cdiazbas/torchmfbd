import importlib
import pkgutil
import numpy as np
import torch
import torch.nn.functional as F
import torchmfbd.zern as zern
import torchmfbd.util as util
from collections import OrderedDict
from tqdm import tqdm
from skimage.morphology import flood
import scipy.ndimage as nd
try:
    from nvitop import Device
    HAS_NVITOP = True
except ImportError:
    HAS_NVITOP = False
import logging
import warnings
import torchmfbd.kl_modes as kl_modes
import torchmfbd.noise as noise
from torchmfbd.reg_smooth import RegularizationSmooth
from torchmfbd.reg_iuwt import RegularizationIUWT
from torchmfbd.reg_time import RegularizationTime, snr_link_factors, SNR_RULE_DEFAULT_THRESHOLD
from torchmfbd.noise import step_snr, step_power_ratio
import glob
import pathlib
import yaml
import torchmfbd.configuration as configuration
import time
import scipy.optimize as optim
from astropy.io import fits
from einops import rearrange
from torchmfbd.vae import ConvVAE2D
import importlib.resources
from concurrent.futures import ThreadPoolExecutor
try:
    import ncg_optimizer
    NGC_OPTIMIZER = True
except:
    NGC_OPTIMIZER = False
    pass

_FFT_POOL = None


def _chunked_fft(fn, x):
    """fn (fft2 or ifft2, norm='ortho') over the leading axes in chunks on a thread pool."""
    global _FFT_POOL
    n = torch.get_num_threads()
    if _FFT_POOL is None or _FFT_POOL._max_workers != n:
        _FFT_POOL = ThreadPoolExecutor(n)
    flat = x.reshape(-1, *x.shape[-2:])
    parts = list(_FFT_POOL.map(lambda c: fn(c, norm='ortho'), torch.tensor_split(flat, n, dim=0)))
    return torch.cat(parts, dim=0).reshape(*x.shape[:-2], *parts[0].shape[-2:])


class _ParallelFFT2(torch.autograd.Function):
    """
    Orthonormal 2-D FFT computed in chunks on a thread pool, for CPU builds of torch whose FFT
    (pocketfft) runs on one thread. Each 2-D transform is the same call as in torch.fft.fft2, so the
    result is bit-identical; the backward is the chunked ifft2, the adjoint of the unitary transform.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.real_in = not x.is_complex()
        return _chunked_fft(torch.fft.fft2, x)

    @staticmethod
    def backward(ctx, g):
        gx = _chunked_fft(torch.fft.ifft2, g)
        return gx.real if ctx.real_in else gx


def fft2_ortho(x):
    """torch.fft.fft2(x, norm='ortho'), threaded on CPU builds without MKL when the batch is large."""
    if (x.device.type == 'cpu' and not torch.backends.mkl.is_available() and torch.get_num_threads() > 1
            and x.ndim >= 3 and x.numel() // (x.shape[-1] * x.shape[-2]) >= 4 * torch.get_num_threads()):
        return _ParallelFFT2.apply(x)
    return torch.fft.fft2(x, norm='ortho')


class Deconvolution(object):
    def __init__(self, config, add_piston=False):        
        """

        Parameters
        ----------
        config : str or dict
            The configuration for the deconvolution. It can be either a path to a YAML file or a dictionary containing the configuration parameters.
        add_piston : bool, optional
            Whether to include the piston mode in the wavefront expansion. The default is False.
        """
        super().__init__()

        torch.set_default_dtype(torch.float32)
        
        self.logger = logging.getLogger("deconvolution ")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.fft_norm = 'ortho'

        if isinstance(config, dict):
            self.logger.info(f"Using configuration dictionary")
            self.config = config
        else:
            self.logger.info(f"Using configuration file {config}")
            self.config = self.read_config_file(config)

        # Check configuration file for errors
        self.config = configuration._check_config(self.config)
                
        # Check the presence of a GPU
        self.cuda = torch.cuda.is_available()      

        # Check that the GPU compatible
        if self.cuda and HAS_NVITOP:
            if len(Device.all()) == 0:
                self.cuda = False  

        # Ger handlers to later check memory and usage of GPUs
        if self.cuda:
            if self.config['optimization']['gpu'] < 0:
                self.logger.info(f"GPU is available but not used. Computing in cpu")
                self.cuda = False
                self.handle = None
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{self.config['optimization']['gpu']}")
                if HAS_NVITOP:
                    self.handle = Device.all()[self.config['optimization']['gpu']]
                    self.logger.info(f"Computing in {self.handle.name()} (free {self.handle.memory_free() / 1024**3:4.2f} GB) - cuda:{self.config['optimization']['gpu']}")
                    self.initial_memory_used = self.handle.memory_used()
                else:
                    self.handle = None
                    self.logger.info(f"Computing in cuda:{self.config['optimization']['gpu']} (nvitop not installed, memory tracking disabled)")
        elif torch.backends.mps.is_available() and self.config['optimization']['gpu'] >= 0:
            self.logger.info(f"Computing in mps (Apple GPU)")
            self.device = torch.device("mps")
            self.handle = None
        else:
            if torch.backends.mps.is_available():
                self.logger.info(f"MPS is available but not used (optimization.gpu < 0). Computing in cpu")
            elif not HAS_NVITOP:
                self.logger.info(f"nvitop not installed. Computing in cpu")
            else:
                self.logger.info(f"No GPU is available. Computing in cpu")
            self.device = torch.device("cpu")
            self.handle = None

        # self.n_modes = np.sum(np.arange(self.config['psf']['nmax_modes'])+2)
        self.n_modes = self.config['psf']['nmax_modes']
        self.use_jitter = self.config['psf']['jitter']
        self.npix = self.config['images']['n_pixel']
        self.npix_apod = self.config['images']['apodization_border']
        self.remove_gradient_apodization = self.config['images']['remove_gradient_apodization']

        self.psf_model = self.config['psf']['model']

        if self.psf_model.lower() in ['nmf', 'vae'] and self.use_jitter != 'none':
            raise ValueError(f"psf.jitter = {self.use_jitter} is only implemented for the zernike and kl PSF models, not for {self.psf_model}")

        self.loss_type = self.config['optimization']['loss_type']
        self.half_spectrum = self.config['optimization']['half_spectrum']
        
        # Whether to take into account the piston mode
        self.add_piston = add_piston                     
                
        # Generate Hamming window function for WFS correlation
        if (self.npix_apod > 0):
            self.logger.info(f"Using apodization mask with a border of {self.npix_apod} pixels")
            if self.remove_gradient_apodization:
                self.logger.info(f"Removing gradient in apodization")
            else:
                self.logger.info(f"Not removing gradient in apodization")
            win = np.hanning(2*self.npix_apod)
            winOut = np.ones(self.npix)
            winOut[0:self.npix_apod] = win[0:self.npix_apod]
            winOut[-self.npix_apod:] = win[-self.npix_apod:]
            self.window = np.outer(winOut, winOut)
        else:
            self.logger.info(f"No apodization")
            self.window = np.ones((self.npix, self.npix))

        self.window = torch.tensor(self.window.astype('float32')).to(self.device)        
                
        # Learning rates
        self.lr_obj = self.config['optimization']['lr_obj']
        self.lr_modes = self.config['optimization']['lr_modes']
        self.lr_prior = self.config['optimization']['lr_prior']
        if self.loss_type == 'marginal':
            self.stop_psd = self.config['optimization'].get('stop_psd_optimization', 1000)

        # Optional stopping rule (off by default): stop a batch once the relative change of the loss stays
        # below `tolerance` for `patience` iterations, counted only after the annealing has reached all modes
        self.tolerance = self.config['optimization'].get('tolerance', None)
        self.patience = int(self.config['optimization'].get('patience', 3))

        # Inner iterations of one LBFGS step (torch's default 20; one step costs about max_iter + line search closures)
        self.lbfgs_max_iter = int(self.config['optimization'].get('lbfgs_max_iter', 20))

        # Hard bounds of the PSD exponent p (s_u ~ (1 + (v/v0)^2)^(-p/2)): unbounded, p ran to infinity
        # under LBFGS while the loss stayed finite
        self.log_p_bounds = (float(np.log(0.5)), float(np.log(10.0)))

        # Type of Fourier filter for the loss
        if 'filter_loss' in self.config['optimization']:
            self.loss_filter = self.config['optimization']['filter_loss']
        else:
            self.loss_filter = 'wiener'        
        self.logger.info(f"Using {self.loss_filter} filter for loss computation")

        self.enforce_positive_object = self.config['optimization']['enforce_positive_object']        
        if self.enforce_positive_object:
            self.logger.info(f"Enforcing positive object during optimization")

        # Do some output
        self.logger.info(f"Telescope")        
        self.logger.info(f"  * D: {self.config['telescope']['diameter']} cm")
        self.logger.info(f"  * pix: {self.config['images']['pix_size']} arcsec")
        
        # Bookkeeping for objects and diversity
        self.ind_object = []
        self.ind_diversity = []
        self.frames = []
        self.sigma = []
        self.fixed_pattern = []
        self.diversity = []

        self.external_regularizations = []

        if 'show_object_info' in self.config['optimization']:
            self.show_object_info = self.config['optimization']['show_object_info']
        else:
            self.show_object_info = False

        self.simultaneous_sequences = None
        self.infer_object = None

        if 'orthogonalize' in self.config['psf']:
            self.orthogonalize_basis = self.config['psf']['orthogonalize']
        else:
            self.orthogonalize_basis = False

        if 'remove_tt' in self.config['psf']:
            self.remove_tt = self.config['psf']['remove_tt']
        else:
            self.remove_tt = False

        # Wavefront restricted to the directions the data determine better than a Kolmogorov prior of this D/r0
        # (psf.nullspace_d_r0; absent: no restriction). See nullspace_projectors
        self.nullspace_d_r0 = self.config['psf'].get('nullspace_d_r0', None)
        self.nullspace_threshold = float(self.config['psf'].get('nullspace_threshold', 1.0))
        self.nullspace_P_seq = None
        self.nullspace_P_batch = None
        # Pairs of neighbouring sequences (patches) for the coupling regularization (set_neighbours), and the pairs of
        # the batch being fitted, as positions in the batch
        self.coupling_pairs = None
        self.coupling_E_batch = None
        self.kl_variance = None

        if self.remove_tt:
            self.logger.info(f"Tip-tilt will be removed from the modes during optimization")

       
    def _define_basis(self, n_modes=None):

        if n_modes is not None:
            self.n_modes = n_modes

        if self.psf_model.lower() not in ['zernike', 'kl', 'nmf', 'vae']:
            raise ValueError(f"psf model '{self.psf_model}' is not available in the spatially invariant deconvolution (pca is only implemented in DeconvolutionSV)")

        if self.psf_model.lower() in ['zernike', 'kl']:
            
            # Get Noll's n order from the number of modes
            # The summation of the modes needs to fulfill n^2+n-2*(k+1)=0 when no piston is added
            # The summation of the modes needs to fulfill n^2+n-2*k=0 when a piston is added            
            if (self.add_piston):
                self.n_modes += 1
                a = 1.0
                b = 1.0
                c = -2.0 * self.n_modes
            else:            
                a = 1.0
                b = 1.0
                c = -2.0 * (self.n_modes + 1)            
            
            sol1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
            sol2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
            
            n = 0        
            if sol1 > 0.0 and sol1.is_integer():
                n = int(sol1)
            if sol2 > 0.0 and sol2.is_integer():
                n = int(sol2)
            
            if n == 0:
                if sol1 > 0.0:
                    sol = np.floor(sol1)
                if sol2 > 0.0:
                    sol = np.floor(sol2)
                
                if (self.add_piston):
                    k_sol = (sol**2 + sol) / 2.0                    
                else:
                    k_sol = (sol**2 + sol) / 2.0 - 1
                                        
                raise Exception(f"Number of modes {self.n_modes} do not cover a full radial degree. Closest value : {k_sol}")

            self.noll_max = n   

        self.pupil = [None] * self.n_o
        self.basis = [None] * self.n_o
        self.basis_shifted = [None] * self.n_o
        self.defocus_basis = [None] * self.n_o
        self.rho = [None] * self.n_o
        self.f_x = [None] * self.n_o
        self.f_y = [None] * self.n_o
        self.s_u = [None] * self.n_o
        self.diffraction_limit = [None] * self.n_o
        self.cutoff = [None] * self.n_o
        self.image_filter = [None] * self.n_o

        # First locate unique wavelengths. We will use the same basis for the same wavelength
        ind_wavelengths = []
        unique_wavelengths = []

        for i in range(self.n_o):
            self.cutoff[i] = self.config[f'object{i+1}']['cutoff']
            self.image_filter[i] = self.config[f'object{i+1}']['image_filter']
            # s_u_joint is the object power spectrum level for 64-px patches. The unnormalised DFT of an object
            # has power proportional to npix^2 (and the ortho OTF(0) is 1/npix), so the level used here scales
            # with npix^2, and a configured value means the same at every patch size
            self.s_u[i] = self.s_u_joint_value(i) * (self.npix / 64.0)**2
            w = self.config[f'object{i+1}']['wavelength']
            if w not in unique_wavelengths:
                unique_wavelengths.append(w)
            
            ind_wavelengths.append(unique_wavelengths.index(w))

        # The VAE PSF does not depend on the wavelength: objects at different wavelengths would share it
        if self.psf_model.lower() == 'vae' and len(unique_wavelengths) > 1:
            raise ValueError(f"psf model vae gives the same PSF at every wavelength; objects at {unique_wavelengths} A need a wavefront model (zernike or kl)")

        # Normalize wavelengths to scale basis
        unique_wavelengths = np.array(unique_wavelengths).astype('float32')
        normalized_wavelengths = unique_wavelengths / np.max(unique_wavelengths)

        # Now iterate over all unique wavelengths and associate the basis
        # to the corresponding object
        for i in range(len(unique_wavelengths)):

            wavelength = unique_wavelengths[i]

            # Compute the overfill to properly generate the PSFs from the wavefronts
            overfill = util.psf_scale(wavelength, 
                                    self.config['telescope']['diameter'], 
                                    self.config['images']['pix_size'])

            if (overfill < 1.0):
                raise Exception(f"The pixel size is not small enough to model a telescope with D={self.config['telescope']['diameter']} cm")

            # Compute telescope aperture
            pixel_size_pupil = self.config['telescope']['diameter'] / self.npix
            pupil = util.aperture(npix=self.npix, 
                            cent_obs = self.config['telescope']['central_obscuration'] / self.config['telescope']['diameter'], 
                            spider=self.config['telescope']['spider'] / pixel_size_pupil, 
                            overfill=overfill)
            
            # Obtain defocus basis for phase diversity
            defocus_basis = self.get_defocus_basis(overfill=overfill)
                        
            # PSF model parameterized with the wavefront
            if (self.psf_model.lower() in ['zernike', 'kl', 'nmf']):
                            
                if (self.psf_model.lower() not in ['zernike', 'kl', 'nmf']):
                    raise Exception(f"Unknown basis {self.basis}. Use 'zernike' or 'kl' for wavefront expansion")
            
                if (self.psf_model.lower() == 'zernike'):

                    self.logger.info(f"PSF model: wavefront expansion in Zernike modes")
                    
                    found, filename = self.find_basis_wavefront('zernike', self.n_modes, int(wavelength))

                    # Define Zernike modes        
                    if found:
                        self.logger.info(f"Loading precomputed Zernike {filename}")
                        tmp = np.load(f"{filename}")
                        basis = tmp['basis']           
                    else:                
                        self.logger.info(f"Computing Zernike modes {filename}")
                        basis = self.precalculate_zernike(overfill=overfill)

                        # Add piston mode if needed
                        if (self.add_piston):
                            basis = np.concatenate([pupil * np.ones((1, self.npix, self.npix)), basis[0:self.n_modes, :, :]], axis=0)
                            self.n_modes += 1

                        # Orthogonalize the Zernike modes if needed
                        if self.orthogonalize_basis:
                            self.logger.info(f"Orthogonalizing Zernike modes")
                            basis = util.orthogonalize(basis, pupil)
                            self.logger.info(f"  * Orthogonalization done")

                        np.savez(f"{filename}", basis=basis)

                if (self.psf_model.lower() == 'kl'):

                    self.logger.info(f"PSF model: wavefront expansion in KL modes")

                    found, filename = self.find_basis_wavefront('kl', self.n_modes, int(wavelength))

                    if found:
                        self.logger.info(f"Loading precomputed KL basis: {filename}")
                        tmp = np.load(f"{filename}")
                        basis = tmp['basis']                
                        self.kl_variance = tmp['variance'] if 'variance' in tmp else None
                    else:
                        self.logger.info(f"Computing KL modes {filename}")
                        self.kl = kl_modes.KL()
                        
                        basis = self.kl.precalculate(npix_image = self.npix, 
                                        n_modes_max = self.n_modes,                                 
                                        overfill=overfill)
                        
                        # Add piston mode if needed                        
                        if (self.add_piston):
                            basis = np.concatenate([pupil * np.ones((1, self.npix, self.npix)), basis[0:self.n_modes, :, :]], axis=0)
                            self.n_modes += 1

                        # Orthogonalize the KL modes if needed
                        if self.orthogonalize_basis:
                            self.logger.info(f"Orthogonalizing KL modes")
                            basis = util.orthogonalize(basis, pupil)
                            self.logger.info(f"  * Orthogonalization done")

                        np.savez(f"{filename}", basis=basis, variance=self.kl.varKL)
                        self.kl_variance = self.kl.varKL
                
                if (self.psf_model.lower() == 'nmf'):
                    
                    self.logger.info(f"PSF model: PSF expansion in NMF modes")

                    self.logger.info(f"Loading precomputed NMF basis: {self.config['psf']['filename']}")
                    f = np.load(self.config['psf']['filename'])

                    n_psf = int(np.sqrt(f['basis'].shape[1]))

                    if f['basis'].shape[0] < self.n_modes:
                        raise ValueError(f"The NMF basis file has {f['basis'].shape[0]} modes but psf.nmax_modes requests {self.n_modes}")

                    basis = f['basis'][0:self.n_modes, :].reshape((self.n_modes, n_psf, n_psf))

                    # Crop or zero-pad to npix keeping the centre pixel n_psf//2 at npix//2
                    if n_psf >= self.npix:
                        i0 = n_psf//2 - self.npix//2
                        basis = basis[:, i0:i0 + self.npix, i0:i0 + self.npix]
                    else:
                        left = self.npix//2 - n_psf//2
                        right = self.npix - n_psf - left
                        basis = np.pad(basis, ((0, 0), (left, right), (left, right)), mode='constant')
                                    
                pupil = torch.tensor(pupil.astype('float32')).to(self.device)
                basis = torch.tensor(basis[0:self.n_modes, :, :].astype('float32')).to(self.device)
                defocus_basis = torch.tensor(defocus_basis.astype('float32')).to(self.device)

                # Following van Noort et al. (2005) we normalize the basis to the maximum wavelength
                # so that the wavefront is given in radians
                if (self.psf_model.lower() in ['zernike', 'kl']):
                    basis /= normalized_wavelengths[i]
                    defocus_basis /= normalized_wavelengths[i]
                elif self.psf_model.lower() == 'nmf':
                    # Read the exponent from the YAML config (defaults to 1.0)
                    basis_power = self.config['psf'].get('basis_power', 1.0)
                    
                    if basis_power != 1.0:
                        self.logger.info(f"  * Applying 'fat wing' power: {basis_power}")
                        # Apply power (with epsilon to prevent numerical issues at exactly zero)
                        basis = torch.pow(basis + 1e-12, basis_power)
                    
                    # Normalize each NMF basis function to have unit area
                    basis = basis / (torch.sum(basis, dim=[-1, -2], keepdim=True) + 1e-12)

                if (self.add_piston):
                    self.logger.info(f"Adding piston mode...")
                
                self.logger.info(f"  * Using {self.n_modes} modes...")

            if (self.psf_model.lower() == 'vae'):                
                self.logger.info(f"PSF model: PSF expansion in VAE modes")
               
# Get a Traversable path object to the file
                filename = importlib.resources.files("torchmfbd").joinpath("vae", "nonlinear_matryoshka_z_200_beta_0.1.pth")

                # The checkpoint holds the decoder weights and the PSF statistics only (tensors, ints and a
                # list), so it loads with weights_only=True; it is read once per instance
                if getattr(self, '_vae_checkpoint', None) is None:
                    self._vae_checkpoint = torch.load(filename, map_location=lambda storage, loc: storage, weights_only=True)
                checkpoint = self._vae_checkpoint

                self.x_vae_mean = torch.as_tensor(checkpoint['x_mean'], dtype=torch.float32).to(self.device)
                self.x_vae_std = torch.as_tensor(checkpoint['x_std'], dtype=torch.float32).to(self.device)
                latent_dim = checkpoint['latent_dim']
                n_pixel = checkpoint['n_pixel']
                nested_dims = checkpoint['nested_dims']
                self.vae_latent_dim = latent_dim

                if self.n_modes > latent_dim:
                    raise ValueError(f"psf.nmax_modes = {self.n_modes} but the VAE latent space has {latent_dim} dimensions")

                # Model and Optimizer initialization
                self.model_vae = ConvVAE2D(in_channels=1, img_size=n_pixel, latent_dim=latent_dim, nested_dims=nested_dims).to(self.device)

                # Only the decoder is stored: the encoder is not used to generate PSFs
                missing, unexpected = self.model_vae.load_state_dict(checkpoint['state_dict'], strict=False)
                if unexpected or any(k.startswith('decoder') for k in missing):
                    raise RuntimeError(f"VAE checkpoint does not match the decoder: missing {missing}, unexpected {unexpected}")
                self.model_vae.eval()

                # The decoder is a fixed prior: no gradients for its 1.5 M weights
                self.model_vae.requires_grad_(False)

                self.logger.info(f"VAE is trained for PSFs of size {n_pixel}x{n_pixel} pixels with latent dimension {latent_dim}")
                if n_pixel > self.npix:
                    self.logger.info(f"PSFs will be cropped to generate PSFs of size {self.npix}x{self.npix} pixels")
                if n_pixel < self.npix:
                    self.logger.info(f"PSFs will be padded to generate PSFs of size {self.npix}x{self.npix} pixels")
                
                self.logger.info(f"Number of VAE model parameters: {sum(p.numel() for p in self.model_vae.parameters())}")
            
            # Output jitter
            if self.use_jitter == 'slow' or self.use_jitter == 'fast':
                self.logger.info(f"PSF model: including {self.use_jitter} jitter")
                
            # Compute the diffraction limit and the frequency grid
            # Frequency is given in units of the cutoff frequency
            cutoff = self.config['telescope']['diameter'] / (wavelength * 1e-8) / 206265.0
            freq = np.fft.fftfreq(self.npix, d=self.config['images']['pix_size']) / cutoff
            
            f_x, f_y = np.meshgrid(freq, freq)
            rho = np.sqrt(f_x ** 2 + f_y ** 2).astype('float32')
            rho = torch.tensor(rho.astype('float32')).to(self.device)

            # Fourier coordinates for jitter OTF given in pixel
            f = np.fft.fftfreq(self.npix)
            f_x, f_y = np.meshgrid(f, f)
            f_x = torch.tensor(f_x.astype('float32')).to(self.device)
            f_y = torch.tensor(f_y.astype('float32')).to(self.device)

            x = np.linspace(-1, 1, self.npix)
            y = np.linspace(-1, 1, self.npix)            
            self.X = torch.tensor(x.astype('float32')).to(self.device)
            self.Y = torch.tensor(y.astype('float32')).to(self.device)
            self.X = self.X.view(1, 1, self.npix, 1)            
            self.Y = self.Y.view(1, 1, 1, self.npix)
            self.K = 50
            self.sigma_alias = 0.02
            self.tau = torch.linspace(0, 1, self.K).to(self.device)
            
            diffraction_limit = wavelength * 1e-8 / self.config['telescope']['diameter'] * 206265.0

            self.logger.info(f"Wavelength {i} ({wavelength} A)")
            self.logger.info(f"  * Diffraction: {diffraction_limit} arcsec")
            self.logger.info(f"  * Diffraction (x1.22): {1.22 * diffraction_limit} arcsec")

            for j in range(self.n_o):
                if ind_wavelengths[j] == i:

                    if self.psf_model.lower() in ['zernike', 'kl', 'nmf']:                        
                        self.pupil[j] = pupil
                        self.basis[j] = basis
                        self.defocus_basis[j] = defocus_basis
                    self.rho[j] = rho
                    self.f_x[j] = f_x
                    self.f_y[j] = f_y
                    self.diffraction_limit[j] = diffraction_limit

        return
    
    def set_regularizations(self):

        # Regularization parameters
        self.logger.info(f"Regularizations")
        self.regularization = []
        self.index_regularization = {
            'tiptilt': [],
            'modes': [],
            'object': []
        }
        
        loop = 0

        # lambda_t of filter_loss wiener_time*; 0 unless a `time` entry sets it (lambda = 0 is a valid control run)
        # The type of a regularization is the prefix of its key: smooth*, iuwt*, time*
        def kind(k):
            for prefix in ('smooth', 'iuwt', 'time', 'coupling'):
                if k.startswith(prefix):
                    return prefix
            raise ValueError(f"Unknown regularization '{k}': the key must start with smooth, iuwt, time or coupling")

        self.time_regularization = 0.0
        self.time_mask_window = 3
        # SNR rule of the time weight (0: one weight for every link)
        self.time_snr_threshold = SNR_RULE_DEFAULT_THRESHOLD
        self.time_snr_reference = 'window'
        self.time_snr_source = 'frames'
        # Coupling of the wavefronts of neighbouring patches (needs set_neighbours); 0 unless a `coupling` entry sets it
        self.coupling_lambda = 0.0
        self.coupling_modes = 10
        for k, v in self.config['regularization'].items():
            if kind(k) == 'coupling':
                self.coupling_lambda = float(v['lambda'])
                self.coupling_modes = int(v.get('n_modes', 10))
                if self.coupling_lambda < 0 or self.coupling_modes < 1:
                    raise ValueError(f"regularization.{k} needs lambda >= 0 and n_modes >= 1")
            if kind(k) == 'time':
                self.time_regularization = float(v['lambda'])
                self.time_mask_window = int(v.get('mask_window', 3))
                if self.time_mask_window < 1 or self.time_mask_window % 2 == 0:
                    raise ValueError(f"regularization.{k}.mask_window must be a positive odd number of time steps, got {self.time_mask_window}")
                self.time_snr_threshold = float(v.get('snr_threshold', SNR_RULE_DEFAULT_THRESHOLD))
                self.time_snr_reference = str(v.get('snr_reference', 'window'))
                self.time_snr_source = str(v.get('snr_source', 'frames'))
                if self.time_snr_source not in ('frames', 'power'):
                    raise ValueError(f"regularization.{k}.snr_source must be 'frames' or 'power', got '{self.time_snr_source}'")
                if self.time_snr_threshold < 0:
                    raise ValueError(f"regularization.{k}.snr_threshold must be >= 0 (0 switches the SNR rule off), got {self.time_snr_threshold}")
                if self.time_snr_reference not in ('window', 'fixed'):
                    raise ValueError(f"regularization.{k}.snr_reference must be 'window' or 'fixed', got '{self.time_snr_reference}'")
                if 'snr_threshold' in v and self.time_snr_threshold > 0 and self.config['optimization'].get('filter_loss') == 'wiener_time_contrast':
                    raise ValueError(f"regularization.{k}.snr_threshold needs filter_loss wiener_time: wiener_time_contrast already weights the links by the contrast")

        for k, v in self.config['regularization'].items():            
            if v['lambda'] != 0.0:
                
                if kind(k) == 'smooth':                
                    tmp = RegularizationSmooth(lambda_reg=v['lambda'], variable=v['variable'])
                if kind(k) == 'iuwt':
                    tmp = RegularizationIUWT(lambda_reg=v['lambda'], variable=v['variable'], nbands=v['nbands'], n_pixel=self.npix)
                if kind(k) not in ('time', 'coupling'):
                    self.regularization.append(tmp.to(self.device))
                    self.index_regularization[v['variable']].append(loop)
                    tmp.print()
                    loop += 1

        self.logger.info(f"External regularizations")
        for reg in self.external_regularizations:
            self.regularization.append(reg.to(self.device))
            self.index_regularization[reg.variable].append(loop)
            reg.print()
            loop += 1

    def set_neighbours(self, pairs):
        """
        Declare which sequences are neighbouring patches of the same field, for the `coupling` regularization:
        lambda * sum over the pairs (p, q) and the frames of ||theta_p - theta_q||^2 over the first n_modes wavefront
        coefficients (tip-tilt referred to its mean over the frames), in the normalisation of the joint loss (divided by the
        number of sequences of the batch and by (n_pixel - 1)^2). Neighbouring patches see almost the same low-order
        seeing; the term lets them differ where the data ask for it.

        Parameters
        ----------
        pairs : list of (int, int)
            Indices of the sequences as added with add_frames (with flatten_sequences=True), e.g. Patchify4D.neighbours().
            Only pairs whose two patches are fitted in the same batch are coupled: give simultaneous_sequences at least
            the number of patches.
            With a time dimension the pairs still name patches; each is coupled within every time step, not across steps.
        """
        self.coupling_pairs = [(int(p), int(q)) for p, q in pairs]

    def add_external_regularizations(self, external_regularization):        
        """
        Adds external regularizations to the model.

        Parameters:
        -----------

        external_regularization : callable
            A function or callable object that applies the external regularization.
        lambda_reg : float
            The regularization strength parameter.
        variable : str
            The name of the variable to which the regularization is applied.
        **kwargs : dict
            Additional keyword arguments to pass to the external_regularization function.

        """

        # Regularization parameters
        self.logger.info(f"External regularization")
        
        self.external_regularizations.append(external_regularization)
        
    def read_config_file(self, filename):
        """
        Read a configuration file in YAML format.

        Parameters:
        -----------
        filename : str
            The name of the configuration file.

        Returns:
        --------
        dict
            A dictionary containing the configuration parameters.
        """

        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        
        return config

    def find_basis_wavefront(self, basis, nmax, wavelength):

        p = pathlib.Path('basis/')
        p.mkdir(parents=True, exist_ok=True)

        files = glob.glob(f"basis/{basis}_{int(self.config['telescope']['diameter'])}cm_{self.config['images']['n_pixel']}px_{wavelength}A_*.npz")

        if len(files) == 0:
            return False, f"basis/{basis}_{int(self.config['telescope']['diameter'])}cm_{self.npix}px_{wavelength}A_{nmax}.npz"
        
        nmodes = []

        for f in files:
            n = int(f.split('_')[-1].split('.')[0])
            if n >= nmax:
                nmodes.append(n)
                self.logger.info(f"Found basis file with {n} modes that can be used for {nmax} modes")

        if len(nmodes) == 0:
            return False, f"basis/{basis}_{int(self.config['telescope']['diameter'])}cm_{self.npix}px_{wavelength}A_{nmax}.npz"
        
        filename = f"basis/{basis}_{int(self.config['telescope']['diameter'])}cm_{self.npix}px_{wavelength}A_{min(nmodes)}.npz"
        
        return True, filename
        
    def precalculate_zernike(self, overfill):
        """
        Precalculate Zernike polynomials for a given overfill factor.
        This function computes the Zernike polynomials up to `self.n_modes` and 
        returns them in a 3D numpy array. The Zernike polynomials are calculated 
        over a grid defined by `self.npix` and scaled by the `overfill` factor.
        Parameters:
        -----------
        overfill : float
            The overfill factor used to scale the radial coordinate `rho`.
        Returns:
        --------
        Z : numpy.ndarray
            A 3D array of shape (self.n_modes, self.npix, self.npix) containing 
            the precalculated Zernike polynomials. Each slice `Z[mode, :, :]` 
            corresponds to a Zernike polynomial mode.
        """
        
        Z_machine = zern.ZernikeNaive(mask=[])
        x = np.linspace(-1, 1, self.npix)
        xx, yy = np.meshgrid(x, x)
        rho = overfill * np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        aperture_mask = rho <= 1.0

        Z = np.zeros((self.n_modes, self.npix, self.npix))

        noll_Z = 2 + np.arange(self.n_modes)
        
        for mode in tqdm(range(self.n_modes)):
                                                
            jz = noll_Z[mode]
            n, m = zern.zernIndex(jz)
            Zmode = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
            Z[mode, :, :] = Zmode * aperture_mask
                
        return Z
    
    def get_defocus_basis(self, overfill):
        """
        Precalculate Zernike polynomials for a given overfill factor.
        This function computes the Zernike polynomials up to `self.n_modes` and 
        returns them in a 3D numpy array. The Zernike polynomials are calculated 
        over a grid defined by `self.npix` and scaled by the `overfill` factor.
        Parameters:
        -----------
        overfill : float
            The overfill factor used to scale the radial coordinate `rho`.
        Returns:
        --------
        Z : numpy.ndarray
            A 3D array of shape (self.n_modes, self.npix, self.npix) containing 
            the precalculated Zernike polynomials. Each slice `Z[mode, :, :]` 
            corresponds to a Zernike polynomial mode.
        """
        
        Z_machine = zern.ZernikeNaive(mask=[])
        x = np.linspace(-1, 1, self.npix)
        xx, yy = np.meshgrid(x, x)
        rho = overfill * np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        aperture_mask = rho <= 1.0
        
        defocus = Z_machine.Z_nm(2, 0, rho, theta, True, 'Jacobi')
                
        return defocus * aperture_mask
    
    def compute_annealing(self, modes, n_iterations):
        """
        Annealing function
        We start with 2 modes and end with all modes but we give steps of the number of
        Zernike modes for each n

        Args:
            annealing (_type_): _description_
            n_iterations (_type_): _description_

        Returns:
            _type_: _description_
        """        
        
        anneal = np.zeros(n_iterations, dtype=int)
        
        # Annealing schedules          
        if self.config['annealing']['type'] == 'linear':
            self.logger.info(f"Adding modes using linear schedule")
            for i in range(n_iterations):
                if (i < self.config['annealing']['start_pct'] * n_iterations):
                    anneal[i] = modes[0]
                elif (i > self.config['annealing']['end_pct'] * n_iterations):
                    anneal[i] = modes[-1]
                else:
                    x0 = self.config['annealing']['start_pct'] * n_iterations
                    x1 = self.config['annealing']['end_pct'] * n_iterations
                    y0 = 0
                    y1 = len(modes)-1
                    index = np.clip((y1 - y0) / (x1 - x0) * (i - x0) + y0, y0, y1)
                    anneal[i] = modes[int(index)]

        if self.config['annealing']['type'] == 'sigmoid':
            self.logger.info(f"Adding modes using sigmoid schedule")
            a = 7
            b = -5
            x = np.linspace(0, 1, n_iterations)            
            anneal = (len(modes) - 1) * (util.sigmoid(a) - util.sigmoid(a + x * (b-a)) ) / ( util.sigmoid(a) - util.sigmoid(b) )
            
            anneal = (anneal + 0.1).astype(int)            
            anneal = modes[anneal]

        if self.config['annealing']['type'] == 'none':
            self.logger.info(f"All modes always active")
            anneal = np.ones(n_iterations, dtype=int) * modes[-1]

        return anneal
    
    def compute_diffraction_masks(self):
        """
        Compute the diffraction masks for the given dimensions and store them as class attributes.
        Args:
            n_x (int): The number of pixels in the x-dimension.
            n_y (int): The number of pixels in the y-dimension.
        Attributes:
            mask_diffraction (numpy.ndarray): A 3D array of shape (n_o, n_x, n_y) containing the diffraction masks.
            mask_diffraction_th (torch.Tensor): A tensor containing the diffraction masks, converted to float32 and moved to the specified device.
            mask_diffraction_shift (numpy.ndarray): A 3D array of shape (n_o, n_x, n_y) containing the FFT-shifted diffraction masks.
        """
        
        # Compute the diffraction masks and convert to tensor
        self.mask_diffraction = [None] * self.n_o
        self.mask_diffraction_th = [None] * self.n_o
        self.mask_diffraction_shift = [None] * self.n_o
        
        for i in range(self.n_o):
            
            self.mask_diffraction[i] = torch.zeros_like(self.rho[i])
            ind = torch.where(self.rho[i] <= self.cutoff[i][0])
            self.mask_diffraction[i][ind[0], ind[1]] = 1.0

            ind = torch.where(self.rho[i] > self.cutoff[i][1])
            self.mask_diffraction[i][ind[0], ind[1]] = 0.0

            ind = torch.where((self.rho[i] > self.cutoff[i][0]) & (self.rho[i] <= self.cutoff[i][1]))

            # self.mask_diffraction[i][ind[0], ind[1]] = 1.0 - (self.rho[i][ind[0], ind[1]] - self.cutoff[i][0]) / (self.cutoff[i][1] - self.cutoff[i][0])
            self.mask_diffraction[i][ind[0], ind[1]] = torch.cos(np.pi / 2.0 * (self.rho[i][ind[0], ind[1]] - self.cutoff[i][0]) / (self.cutoff[i][1] - self.cutoff[i][0]))

            self.mask_diffraction_th[i] = self.mask_diffraction[i].to(self.device).float()
            
            # Shifted mask used for the Lofdahl & Scharmer filter
            self.mask_diffraction_shift[i] = np.fft.fftshift(self.mask_diffraction[i].cpu().numpy())
                     
    def compute_psfs(self, modes, diversity, jitter=None, half=False):
        """
        Compute the Point Spread Functions (PSFs) from the given modes.
        Parameters:
            
            modes (torch.Tensor): A tensor of shape (batch_size, num_modes, height, width) representing the modes.
            diversity (torch.Tensor): A tensor of shape (batch_size, num_diversity, height, width) representing the diversity.
            jitter (torch.Tensor, optional): A tensor representing the jitter. Default is None.

        Returns:
        tuple: A tuple containing:
            - wavefront (torch.Tensor): The computed wavefronts from the estimated modes.
            - psf_norm (torch.Tensor): The normalized PSFs.
            - otf (torch.Tensor): The FFT of the normalized PSFs.
        """
        

        n_seq, n_f, n_active = modes.shape

        if self.nullspace_P_batch is not None:
            modes = self.restrict_modes(modes)
                                        
        psf_norm = [None] * self.n_o
        otf = [None] * self.n_o
        
        if self.use_jitter == 'slow' and n_active == self.n_modes:
                
            # 1. Extract P1 and P2 and reshape for broadcasting
            # Slice shape: [n_seq, nf, 2] -> Unsqueeze to [n_seq, nf, 1, 2]
            P1 = jitter[:, :, 0, :]  # Shape [n_seq, nf, 2]
            P2 = jitter[:, :, 1, :]  # Shape [n_seq, nf, 2]
            
            # Initialize jitter PSF with zeros
            jitter_psf = torch.zeros((n_seq, n_f, self.npix, self.npix), device=self.device)

            for K in range(self.K):

                t = self.tau[K]

                # Compute curve point for step K using the quadratic Bézier formula
                p_k = 2 * (1 - t) * t * P1 + (t ** 2) * P2  # Shape [n_seq, nf, 2]

                # Split into x and y components
                p_x = p_k[..., 0].unsqueeze(-1).unsqueeze(-1)
                p_y = p_k[..., 1].unsqueeze(-1).unsqueeze(-1)

                # 4. Compute 1D separable distances
                # dist_x_sq shape: [n_seq, nf, nx, 1]
                # dist_y_sq shape: [n_seq, nf, 1, ny]
                dist_x_sq = (self.X - p_x) ** 2
                dist_y_sq = (self.Y - p_y) ** 2
                
                # 5. Compute 1D Gaussians
                psf_x = torch.exp(-dist_x_sq / (2 * self.sigma_alias ** 2))
                psf_y = torch.exp(-dist_y_sq / (2 * self.sigma_alias ** 2))

                # 6. Outer product (via broadcasting) creates the [nx, ny] 2D Gaussian
                # Add to the accumulator
                jitter_psf += (psf_x * psf_y)

            jitter_psf /= self.K  # Average over K steps

            # NNormalize to ensure energy conservation
            psf_sum = torch.sum(jitter_psf, dim=(-1, -2), keepdim=True)
            jitter_psf = jitter_psf / (psf_sum + 1e-8)
            jitter_psf = torch.fft.fftshift(jitter_psf, dim=(-2, -1))
            
            # Compute the jitter OTF
            jitter_otf = torch.fft.fft2(jitter_psf)

        same_as_first = getattr(self, '_psf_same_as_first', None)

        for i in range(self.n_o):            

            # An object seen through the same wavefront, pupil, basis and diversity as the first has its PSFs and OTFs:
            # computed once (_deconvolve_pass decides it for each batch). The two objects of a wide band + line fit
            # at one wavelength spend 19 % less time; the gradient is summed before the PSF chain instead of after, so
            # the result changes at the rounding level (12 two-object 49-step series with a truth: mean object error
            # -0.10 % wide band, +0.05 % line, worst series +1.01 %, the spread of a T4 against a V100S, 2026-11-22)
            if same_as_first is not None and same_as_first[i]:
                psf_norm[i], otf[i] = psf_norm[0], otf[0]
                continue

            # Gaussian jitter OTF; f_x, f_y are in cycles per pixel
            if self.use_jitter == 'fast' and n_active > 20:
                sigma_x = torch.exp(jitter[:, :, 0])
                sigma_y = torch.exp(jitter[:, :, 1])
                rho_xy = torch.tanh(jitter[:, :, 2]) * sigma_x * sigma_y

                tx = self.f_x[i][None, None, :, :]**2 * sigma_x[:, :, None, None]**2 
                ty = self.f_y[i][None, None, :, :]**2 * sigma_y[:, :, None, None]**2
                txy = 2 * self.f_x[i][None, None, :, :] * self.f_y[i][None, None, :, :] * rho_xy[:, :, None, None]

                jitter_otf = torch.exp(-2 * np.pi**2 * (tx + ty + txy))

            # Compute wavefronts from estimated modes                
            wavefront = torch.einsum('ijk,klm->ijlm', modes, self.basis[i][0:n_active, :, :])
            
            # Reuse the same wavefront per object but add the diversity
            wavef = []
            for j in self.init_frame_diversity[i]:                
                div = diversity[i][:, j:j+n_f, None, None] * self.defocus_basis[i][None, None, :, :]
                wavef.append(wavefront + div)
            
            wavef = torch.cat(wavef, dim=1)
            
            # Compute the complex phase
            phase = self.pupil[i][None, None, :, :] * torch.exp(1j * wavef)

            # Compute FFT of the pupil function and compute autocorrelation
            ft = fft2_ortho(phase) if self.fft_norm == 'ortho' else torch.fft.fft2(phase, norm=self.fft_norm)
            psf = (torch.conj(ft) * ft).real
            
            # Normalize PSF        
            psf_norm[i] = psf / torch.sum(psf, [-1, -2], keepdim=True)

            # FFT of the PSF (half plane for the half-spectrum loss: the PSF is real)
            if half:
                otf[i] = torch.fft.rfft2(psf_norm[i], norm=self.fft_norm)
            else:
                otf[i] = fft2_ortho(psf_norm[i]) if self.fft_norm == 'ortho' else torch.fft.fft2(psf_norm[i], norm=self.fft_norm)

            # If using jitter, multiply the OTFs. Not in place: on CPU the OTF can come from the threaded FFT,
            # a custom autograd function whose output autograd does not allow to be modified
            if self.use_jitter == 'fast' and n_active > 20:
                otf[i] = otf[i] * jitter_otf

            if self.use_jitter == 'slow' and n_active == self.n_modes:
                otf[i] = otf[i] * jitter_otf
                
        return psf_norm, otf
    
    def compute_psfs_nmf(self, modes, shift_x=None, shift_y=None):
        """
        Compute the Point Spread Functions (PSFs) from the given modes.
        Parameters:
        modes (torch.Tensor): A tensor of shape (batch_size, num_modes, height, width) representing the modes.
        Returns:
        tuple: A tuple containing:
            - wavefront (torch.Tensor): The computed wavefronts from the estimated modes.
            - psf_norm (torch.Tensor): The normalized PSFs.
            - otf (torch.Tensor): The FFT of the normalized PSFs.
        """
                
        n_active = modes.shape[2]
                                        
        psf_norm = [None] * self.n_o
        otf = [None] * self.n_o
        
        # Enforce non-negativity of NMF coefficients with scale from config
        scale = self.config['optimization'].get('softplus_scale', 1.0)
        modes_nn = F.softplus(modes * scale) / scale

        for i in range(self.n_o):

            # PSF from the estimated modes, with the basis already shifted to the origin
            if self.basis_shifted[i] is None:
                self.basis_shifted[i] = torch.fft.fftshift(self.basis[i], dim=[-2, -1])
            psf = torch.einsum('ijk,klm->ijlm', modes_nn, self.basis_shifted[i][0:n_active, :, :])
                        
            # Normalize PSF (with epsilon to avoid division by zero)        
            psf_norm[i] = psf / (torch.sum(psf, [-1, -2], keepdim=True) + 1e-12)

            # FFT of the PSF
            otf[i] = fft2_ortho(psf_norm[i]) if self.fft_norm == 'ortho' else torch.fft.fft2(psf_norm[i], norm=self.fft_norm)

            # Apply Fourier-domain shift (tip-tilt substitute for NMF)
            # shift_x, shift_y are in pixels; shape: (n_seq,)
            sx = shift_x if shift_x is not None else (self.shift_x if self.config['psf']['shift'] else None)
            sy = shift_y if shift_y is not None else (self.shift_y if self.config['psf']['shift'] else None)
            if sx is not None and sy is not None:
                # Referenced to the first frame of each sequence: a common shift of all PSFs is a shift of the object
                sx = sx - sx[:, 0:1]
                sy = sy - sy[:, 0:1]
                freq_x = torch.fft.fftfreq(self.npix, device=self.device)  # (W,)
                freq_y = torch.fft.fftfreq(self.npix, device=self.device)  # (H,)
                fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')     # (H, W)
                # shift: (n_seq, n_f, 1, 1) for broadcasting over (n_seq, n_f, H, W)
                phase = (-2j * torch.pi * (
                    sx[:, :, None, None] * fx[None, None, :, :] +
                    sy[:, :, None, None] * fy[None, None, :, :]
                ))
                otf[i] = otf[i] * torch.exp(phase)
        
        return psf_norm, otf

    def compute_psfs_vae(self, latent_codes, shift_x=None, shift_y=None):
        """
        Compute the Point Spread Functions (PSFs) from the given latent codes using a VAE model.
        Parameters:
        latent_codes (torch.Tensor): A tensor of shape (batch_size, latent_dim) representing the latent codes.
        Returns:
        tuple: A tuple containing:
            - psf_norm (torch.Tensor): The normalized PSFs.
            - otf (torch.Tensor): The FFT of the normalized PSFs.
        """
        
        n_seq = latent_codes.shape[0]
        
        psf_norm = [None] * self.n_o
        otf = [None] * self.n_o
                
        # Decode latent codes to PSFs using the VAE model
        
        if latent_codes.shape[-1] < self.vae_latent_dim:
            latent_codes = torch.nn.functional.pad(latent_codes, (0, self.vae_latent_dim - latent_codes.shape[-1]))

        latent_codes = rearrange(latent_codes, 'ns nf nz -> (ns nf) nz')  # Reshape for VAE input
        
        decoded_psfs = self.model_vae.decode(latent_codes)  # Shape: (n_seq, 1, H, W)

        decoded_psfs = rearrange(decoded_psfs, '(ns nf) 1 h w -> ns nf h w', ns=n_seq)  # Reshape back to (n_seq, n_f, H, W)

        decoded_psfs = decoded_psfs * (self.x_vae_std + 1e-8) + self.x_vae_mean

        # PSF = exp(log-PSF) normalised to unit sum, as a softmax over the pixels: the maximum is
        # subtracted, so large latents cannot overflow exp()
        h, w = decoded_psfs.shape[-2:]
        decoded_psfs = torch.softmax(decoded_psfs.reshape(decoded_psfs.shape[0], decoded_psfs.shape[1], h * w), dim=-1).reshape(decoded_psfs.shape)

        # The VAE generates PSFs of size 64x64. Crop or zero-pad to npix keeping the centre pixel n//2 at npix//2
        n = decoded_psfs.shape[-1]
        if n > self.npix:
            i0 = n // 2 - self.npix // 2
            decoded_psfs = decoded_psfs[:, :, i0:i0 + self.npix, i0:i0 + self.npix]
        if n < self.npix:
            left = self.npix // 2 - n // 2
            right = self.npix - n - left
            decoded_psfs = F.pad(decoded_psfs, (left, right, left, right))

        # Centre pixel npix//2 to the origin (ifftshift; fftshift differs for odd npix)
        decoded_psfs = torch.fft.ifftshift(decoded_psfs, dim=[-2, -1])
        
        for i in range(self.n_o):
            # Normalize PSF (with epsilon to avoid division by zero)
            psf_norm[i] = decoded_psfs / (torch.sum(decoded_psfs, dim=(-1, -2), keepdim=True) + 1e-12)

            # FFT of the PSF
            otf[i] = fft2_ortho(psf_norm[i]) if self.fft_norm == 'ortho' else torch.fft.fft2(psf_norm[i], norm=self.fft_norm)

            # Apply Fourier-domain shift (tip-tilt substitute for NMF); no shift when none is given
            # shift_x, shift_y are in pixels; shape: (n_seq, n_f)
            if shift_x is None and shift_y is None:
                continue
            if shift_x is None:
                shift_x = torch.zeros_like(shift_y)
            if shift_y is None:
                shift_y = torch.zeros_like(shift_x)
            freq_x = torch.fft.fftfreq(self.npix, device=self.device)  # (W,)
            freq_y = torch.fft.fftfreq(self.npix, device=self.device)  # (H,)
            fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')     # (H, W)
            # shift: (n_seq, n_f, 1, 1) for broadcasting over (n_seq, n_f, H, W)
            phase = (-2j * torch.pi * (
                shift_x[:, :, None, None] * fx[None, None, :, :] +
                shift_y[:, :, None, None] * fy[None, None, :, :]
            ))
            otf[i] = otf[i] * torch.exp(phase)
        
        return psf_norm, otf
    
    def compute_psf_diffraction(self):
        """
        Compute the Point Spread Functions (PSFs) from diffraction
        
        Returns:
        tuple: A tuple containing:
            - psf_norm (torch.Tensor): The normalized PSFs.
            - otf (torch.Tensor): The FFT of the normalized PSFs.
        """
        
        otf = [None] * self.n_o
        psf_norm = [None] * self.n_o
        done = {}

        for i in range(self.n_o):
            # Objects at the same wavelength share the pupil tensor: compute its PSF once
            if id(self.pupil[i]) in done:
                psf_norm[i], otf[i] = done[id(self.pupil[i])]
                continue

            # Compute FFT of the pupil function and compute autocorrelation
            ft = torch.fft.fft2(self.pupil[i], norm=self.fft_norm)
            psf = (torch.conj(ft) * ft).real
            
            # Normalize PSF        
            psf_norm[i] = psf / torch.sum(psf, dim=(-1, -2), keepdim=True)

            # FFT of the PSF
            otf[i] = fft2_ortho(psf_norm[i]) if self.fft_norm == 'ortho' else torch.fft.fft2(psf_norm[i], norm=self.fft_norm)
            done[id(self.pupil[i])] = (psf_norm[i], otf[i])

        return psf_norm, otf
    
    def lofdahl_scharmer_filter(self, Sconj_S, Sconj_I, sigma):
        """
        Applies the Löfdahl-Scharmer filter to the given input tensors.
        Parameters:
        -----------
        Sconj_S : torch.Tensor
            The conjugate of the Fourier transform of the observed image.
        Sconj_I : torch.Tensor
            The conjugate of the Fourier transform of the ideal image.
        Returns:
        --------
        torch.Tensor
            A tensor representing the mask after applying the Löfdahl-Scharmer filter.
        """
        # Both sums are real by construction; dividing in real arithmetic avoids the |den|^2 of a complex
        # division, which underflows to 0 (inf/NaN in H) on devices that flush subnormals, such as MPS
        den = (torch.conj(Sconj_I) * Sconj_I).real
        H = Sconj_S.real / den

        # The sampled pupil's OTF ends before the nominal cutoff: where sum|S|^2 is at roundoff level, H is a
        # ratio of roundoff numbers that the 3 x 3 median would carry into the band, so those frequencies are
        # given no signal
        support = Sconj_S.real > 1e-10 * Sconj_S.real[:, 0:1, 0:1]
        H = torch.where(support, H, torch.full_like(H, float('inf')))
        H = torch.fft.fftshift(H, dim=(-2, -1)).detach().cpu().numpy()
        
        # noise = 1.35 / np.median(H[:, :, 0:10, 0:10], axis=(2,3))

        H = nd.median_filter(H, [1,3,3], mode='wrap')    

        if hasattr(sigma, 'dim') and sigma.dim() > 0:
            if sigma.dim() == 1:
                sigma_1 = sigma[:, None, None].cpu().numpy()
            elif sigma.dim() == 2:
                # per-frame variances (n_seq, n_f): one noise level per sequence
                sigma_1 = torch.mean(sigma, dim=1)[:, None, None].cpu().numpy()
            else:
                sigma_1 = torch.mean(sigma, axis=-1)[:, None, None].cpu().numpy()
        else:
            sigma_1 = sigma.cpu().numpy()
        
        filt = 1.0 - H * sigma_1
        filt[filt < 0.2] = 0.0
        filt[filt > 1.0] = 1.0
                
        nb, nx, ny = filt.shape

        # Zero frequency (the centre after the shift) carries the object's mean, which every frame gives; it is
        # always kept, whatever H says there once the apodization has removed the mean
        filt[:, nx//2, ny//2] = 1.0

        mask = np.zeros_like(filt).astype('float32')

        # The region above the threshold connected to zero frequency. (A flood of 1 - filt with tolerance 0.9
        # around the seed's own value also took the zeroed pixels whenever filt(0) < 0.9.)
        for ib in range(nb):                
            mask[ib, :, :] = flood((filt[ib, :, :] >= 0.2).astype(np.uint8), (nx//2, ny//2), tolerance=0)
            mask[ib, :, :] = np.fft.fftshift(mask[ib, :, :])
        
        return torch.tensor(mask.astype('float32')).to(Sconj_S.device)

    def garrote_filter(self, Sconj_S, Sconj_I, sigma):
        """
        Continuous version of the Löfdahl-Scharmer filter: the gain 1 - sigma^2 sum|S|^2 / |sum S* D|^2,
        with the ratio median-filtered over 3 x 3 frequencies and clipped to [0, 1], is used as a weight instead
        of being thresholded and binarised, and the flood fill from zero frequency is replaced by its
        continuous counterpart (reconstruct_from_zero_frequency). Per frequency the gain is the positive-part
        James-Stein (non-negative garrote) shrinkage of the multi-frame estimate. The result is a continuous
        function of the data, so a small change of the frames gives a small change of the object.

        Parameters:
        -----------
        Sconj_S : torch.Tensor
            sum_j |S_j|^2, shape (n_seq, nx, ny).
        Sconj_I : torch.Tensor
            sum_j S_j^* D_j, shape (n_seq, nx, ny).
        sigma : torch.Tensor or float
            Noise variance: a scalar, one value per sequence or one per frame (averaged over the frames).

        Returns:
        --------
        torch.Tensor
            The gain, shape (n_seq, nx, ny), in [0, 1].
        """
        SS = Sconj_S.real.detach()
        SI2 = (torch.conj(Sconj_I) * Sconj_I).real.detach()

        if hasattr(sigma, 'dim') and sigma.dim() > 0:
            if sigma.dim() == 1:
                s2 = sigma[:, None, None]
            elif sigma.dim() == 2:
                s2 = torch.mean(sigma, dim=1)[:, None, None]
            else:
                s2 = torch.mean(sigma, axis=-1)[:, None, None]
        else:
            s2 = sigma
        if torch.is_tensor(s2):
            s2 = s2.detach().to(SS.device)

        # The sampled pupil's OTF ends before the nominal cutoff: where sum|S|^2 is at roundoff level the
        # ratio is noise, so those frequencies are given no signal before the median
        support = SS > 1e-10 * SS[:, 0:1, 0:1]
        H = torch.where(support, SS / SI2, torch.full_like(SS, float('inf')))

        # 3 x 3 median with periodic boundaries (the spectrum is periodic), by the 19-step min/max network for
        # the median of nine values: exact, and elementwise, so it is fast on every device (torch.median is not on MPS)
        nx, ny = H.shape[-2], H.shape[-1]
        Hp = F.pad(H[:, None, :, :], (1, 1, 1, 1), mode='circular')[:, 0]
        p = [Hp[:, i:i + nx, j:j + ny] for i in range(3) for j in range(3)]
        for a, b in [(1, 2), (4, 5), (7, 8), (0, 1), (3, 4), (6, 7), (1, 2), (4, 5), (7, 8), (0, 3),
                     (5, 8), (4, 7), (3, 6), (1, 4), (2, 5), (4, 7), (4, 2), (6, 4), (4, 2)]:
            p[a], p[b] = torch.minimum(p[a], p[b]), torch.maximum(p[a], p[b])
        H = p[4]

        gain = torch.nan_to_num(torch.clamp(1.0 - H * s2, min=0.0, max=1.0), nan=0.0)

        # Zero frequency carries the object's mean, which every frame gives: always kept, and the
        # reconstruction grows from it
        gain[:, 0, 0] = 1.0

        return self.reconstruct_from_zero_frequency(gain)

    def reconstruct_from_zero_frequency(self, gain):
        """
        Grayscale morphological reconstruction of a gain from the zero frequency (Vincent 1993): for every
        frequency, the largest over 8-connected paths from zero frequency of the smallest gain along the path.
        Islands of gain that are not connected to zero frequency are removed, and gain behind a narrow gap is
        limited by the gap. Thresholding the result at any level gives the flood fill from zero frequency of
        the gain above that level, which is what the Löfdahl-Scharmer filter does at the level 0.2; unlike the
        flood fill, the reconstruction is a continuous function of the gain.

        Parameters:
        -----------
        gain : torch.Tensor
            Gain in the unshifted frequency layout, shape (n_seq, nx, ny).

        Returns:
        --------
        torch.Tensor
            The reconstructed gain, same shape, never above the input.
        """
        nx, ny = gain.shape[-2], gain.shape[-1]
        g = torch.fft.fftshift(gain, dim=(-2, -1))[:, None, :, :]
        r = torch.zeros_like(g)
        r[..., nx // 2, ny // 2] = g[..., nx // 2, ny // 2]

        # Geodesic dilation until nothing changes; the iterates only take values of g and never decrease, so
        # the loop ends exactly, after about the length of the longest path (20-35 steps for 64 px patches)
        for _ in range(nx * ny):
            r_new = torch.minimum(F.max_pool2d(F.pad(r, (1, 1, 1, 1), mode='circular'), 3, stride=1), g)
            if torch.equal(r_new, r):
                break
            r = r_new

        return torch.fft.ifftshift(r[:, 0], dim=(-2, -1))

    def s_u_joint_value(self, obj):
        """
        The configured s_u_joint of an object, a number or 'auto'. With 'auto' the level follows the data:
        250 (sigma / mu / 0.02) max(12 / N, 1) mu^2, with sigma the noise estimate of the frames (mean over frames and
        sequences), mu the mean of the frames and N the frames per sequence; for frames divided by their mean (mu = 1)
        it is 250 (sigma / 0.02) max(12 / N, 1). Measured on the benchmark with
        96- and 128-px patches: the best fixed level grows linearly with the noise and with 12/N, and this rule lands
        on it; with 6 frames it turns a loss into a gain. With 64-px patches it makes the object worse. Above 12 frames
        frames_prior_scale applies 12/N on top, as for a fixed value.
        """
        v = self.config[f'object{obj+1}']['s_u_joint']
        if not (isinstance(v, str) and v.strip().lower() == 'auto'):
            return float(v)
        if obj >= len(self.sigma) or self.sigma[obj] is None:
            raise ValueError(f"object{obj+1}.s_u_joint: auto needs the frames: call add_frames before deconvolve")
        sigma = float(torch.mean(torch.as_tensor(self.sigma[obj], dtype=torch.float32)))
        n_frames = self.frames[obj].shape[1]
        # Scale-free: the rule was measured on frames divided by their mean (mu = 1). For frames in other units the
        # noise is taken relative to the mean and the level, an object power, scales with mu^2
        mu = float(torch.mean(self.frames[obj]))
        return 250.0 * (sigma / mu / 0.02) * max(12.0 / n_frames, 1.0) * mu**2

    def frames_prior_scale(self, obj):
        """
        Factor applied to the joint object prior level s_u for the number of frames of the object.

        The data term of the joint loss grows with the number of frames N while the object prior does not, so with
        many frames the fit drifts along the degenerate direction of MFBD (all PSFs broadened by a common kernel, the
        object sharpened by its inverse). Above 12 frames the prior keeps its weight relative to the data:
        s_u = s_u_joint * 12 / N. At 12 frames or fewer the factor is 1: there the stronger relative prior lowers the
        rms object error but not the PSF error, and the choice is left unchanged.
        """
        n = getattr(self, 'n_frames_per_object', None)
        if n is None or n[obj] <= 12:
            return 1.0
        return 12.0 / n[obj]

    def get_su_s2(self, obj, sigma, pars_s0=None, pars_s2=None, frames_mean=None):
        """
        Transforms the parameters of the s0 function to ensure they are in the correct range.
        Parameters:
        -----------
        pars_s0 : torch.Tensor
            A tensor containing the parameters for the s0 function.
        i : int
            The index of the object for which to compute s_u.
        Returns:
        --------
        tuple: A tuple containing:
            - K (torch.Tensor): The transformed K parameter.
            - v0 (torch.Tensor): The transformed v0 parameter.
            - p (torch.Tensor): The transformed p parameter.
        """
        if self.loss_type == 'marginal':
            if pars_s2 is not None:
                # Upstream format: pars_s0 shape (n_o, 3), pars_s2 shape (n_o, 1)
                K = torch.exp(pars_s0[obj, 0])            
                v0 = torch.exp(pars_s0[obj, 1])
                p = torch.exp(torch.clamp(pars_s0[obj, 2], min=self.log_p_bounds[0], max=self.log_p_bounds[1]))

                v = self.rho[obj]

                # Evaluate the s_u function on the frequency grid.
                # We use the mean of the frames in Fourier space to set the scale of s_u at frequency 0,0
                s_u = K * self.npix / (1.0 + (v/v0)**2)**(p/2.0)
                
                # The DC component of the Fourier transform of the image is equal to sqrt(nx*ny)*mean if norm='ortho'
                # As a consequence, the variance of the noise at frequency (0,0) is given by s_u[0,0] = mean**2 * nx * ny,
                # one value per sequence of the batch
                s_u = s_u[None, :, :].repeat(frames_mean.shape[0], 1, 1)
                s_u[:, 0, 0] = frames_mean**2 * self.npix * self.npix
                
                s2 = torch.exp(pars_s2[obj, 0])
            else:
                # Local format fallback: pars_s0 shape (batch_size, n_o, 4)
                K = torch.exp(pars_s0[:, obj, 0])[:, None, None]
                v0 = torch.exp(pars_s0[:, obj, 1])[:, None, None]            
                p = torch.exp(torch.clamp(pars_s0[:, obj, 2], min=self.log_p_bounds[0], max=self.log_p_bounds[1]))[:, None, None]

                v = self.rho[obj][None, :, :]

                s_u = K / (1.0 + (v/v0)**2)**(p/2.0)

                s_u[:, 0, 0] = frames_mean**2 * self.npix * self.npix
                
                s2 = torch.exp(pars_s0[:, obj, 3])
        else:
            K, v0, p = None, None, None
            s_u = self.s_u[obj] * self.frames_prior_scale(obj) * torch.ones_like(self.rho[obj]).to(self.device)
            if hasattr(sigma[obj], 'dim') and sigma[obj].dim() == 3:
                s2 = torch.mean(sigma[obj]**2, dim=1)
            else:
                s2 = torch.mean(sigma[obj]**2)
        
        return s_u, s2, K, v0, p
            
    def time_window_sums(self, *sums):
        """
        Sum per-step quantities over a window of neighbouring time steps of the same sequence.

        Parameters
        ----------
        *sums : torch.Tensor
            Tensors of shape (n_seq * n_t, nx, ny) in the '(s t)' layout of compute_object, such as
            sum_j gamma |S_j|^2 and sum_j gamma S_j^* D_j.

        Returns
        -------
        list of torch.Tensor
            The same tensors, each step replaced by the sum over steps t - w//2 ... t + w//2 that exist, with w the
            `mask_window` of the `time` regularization (3 by default; 1 gives each step alone).
        """
        half = int(getattr(self, 'time_mask_window', 3)) // 2
        out = []
        for x in sums:
            y = rearrange(x, '(s t) w h -> s t w h', t=self.n_t)
            acc = y.clone()
            for k in range(1, min(half, self.n_t - 1) + 1):
                acc[:, k:] += y[:, :-k]
                acc[:, :-k] += y[:, k:]
            out.append(rearrange(acc, 's t w h -> (s t) w h'))
        return out

    def solve_object_time_evolution(self, S, D, Reg_Space, lambda_s, lambda_t, contrast=None):
        """
        Solves the batched tridiagonal systems for the multi-burst regularized object using PyTorch.
        
        Parameters:
        -----------
        S : torch.Tensor (complex)
            Shape (M, J, Nx, Ny). The OTFs for M bursts, each containing J frames.
        D : torch.Tensor (complex)
            Shape (M, J, Nx, Ny). The Fourier transformed data frames.
        Reg_Space : torch.Tensor (real)
            Shape (Nx, Ny). Precomputed spatial regularization grid |R_s(u, v)|^2.
        lambda_s : float
            Spatial regularization parameter.
        lambda_t : float
            Temporal regularization parameter.
        device : str
            The target computation device ('cuda' or 'cpu').
            
        Returns:
        --------
        O : torch.Tensor (complex)
            Shape (M, Nx, Ny). The regularized objects for all M bursts on the specified device.
        """
        
        # Detect whether we want a variable lambda_t
        # In such case, weight the lambda_t by the inverse of the contrast of each image in the
        # burst, so that worse images are more regularized than better ones.
        lambda_t_variable = False
        if contrast is not None:            
            # The edge (t, t+1) is weighted by both of its ends (geometric mean of their normalised contrasts),
            # so the result does not depend on the direction of time
            contrast_normalized = contrast / torch.min(contrast)
            lambda_t = lambda_t / (contrast_normalized[0:-1] * contrast_normalized[1:])
            lambda_t_variable = True
        
        M, J, Nx, Ny = S.shape
        B = Nx * Ny
        
        # 1. Compute Data Components and RHS Vector
        sum_S_sq_flat = torch.sum(torch.abs(S)**2, dim=1).reshape(M, B)
        RHS = torch.sum(torch.conj(S) * D, dim=1).reshape(M, B)
        reg_s_flat = (lambda_s * Reg_Space).flatten()

        # A single time step has no temporal neighbours: the system is diagonal (same floor as the
        # non-time scharmer estimate, so frequencies without data give 0, not 0/0)
        if M == 1:
            return (RHS / (sum_S_sq_flat + reg_s_flat + 1e-10)).reshape(M, Nx, Ny)

        # Frequencies without data in any time step (sum|S|^2 = 0 beyond the cutoff) make the Neumann
        # system singular; their solution is undefined and masked out later. They are solved as an
        # identity row and set to 0, so the last pivot cannot be 0/0 and NaN cannot reach the object.
        lt_max = float(torch.max(torch.as_tensor(lambda_t))) if lambda_t_variable else float(lambda_t)
        w_freq = torch.sum(sum_S_sq_flat, dim=0)
        no_data = (w_freq <= 1e-6 * lt_max) | (w_freq <= 1e-10 * torch.max(w_freq))
        if torch.any(no_data):
            sum_S_sq_flat = torch.where(no_data[None, :], torch.ones_like(sum_S_sq_flat), sum_S_sq_flat)

        # complex128 where the device has it (MPS does not)
        work = torch.complex64 if S.device.type == 'mps' else torch.complex128
        
        # 2. Build Full Arrays for Diagonals out-of-place
        # Explicitly tracking lower (a) and upper (c) arrays as requested
        if lambda_t_variable:
            a = -lambda_t.unsqueeze(1).expand(M - 1, B).to(work).to(S.device)
            c = -lambda_t.unsqueeze(1).expand(M - 1, B).to(work).to(S.device)

                # Build main diagonal list of rows safely
            d_rows = []
            for m in range(M):
                row = sum_S_sq_flat[m, :] + reg_s_flat
                if m == 0:
                    row = row + lambda_t[0]
                elif m == M - 1:
                    row = row + lambda_t[M - 2]
                else:
                    row = row + lambda_t[m - 1] + lambda_t[m]
                d_rows.append(row)

        else:
            a = torch.full((M - 1, B), -lambda_t, dtype=work, device=S.device)
            c = torch.full((M - 1, B), -lambda_t, dtype=work, device=S.device)
        
            # Build main diagonal list of rows safely
            d_rows = []
            for m in range(M):
                row = sum_S_sq_flat[m, :] + reg_s_flat
                if m == 0 or m == M - 1:
                    row = row + lambda_t
                else:
                    row = row + 2.0 * lambda_t
                d_rows.append(row)

        # 3. Forward Pass using clean out-of-place accumulation lists
        c_prime_list = []
        d_prime_list = []
        
        # Base Row (m = 0)
        d_base = d_rows[0]
        c_prime_list.append(c[0, :] / d_base)
        d_prime_list.append(RHS[0, :] / d_base)
        
        # Intermediate Rows (m = 1 to M-2)
        for m in range(1, M - 1):
            denom = d_rows[m] - a[m - 1, :] * c_prime_list[m - 1]
            c_prime_list.append(c[m, :] / denom)
            d_prime_list.append((RHS[m, :] - a[m - 1, :] * d_prime_list[m - 1]) / denom)
            
        # Last Row (m = M-1)
        if M > 1:
            denom = d_rows[M - 1] - a[M - 2, :] * c_prime_list[M - 2]
            d_prime_list.append((RHS[M - 1, :] - a[M - 2, :] * d_prime_list[M - 2]) / denom)

        # 4. Backward Substitution Pass
        O_list = [None] * M
        O_list[M - 1] = d_prime_list[M - 1]
        
        for m in range(M - 2, -1, -1):
            O_list[m] = d_prime_list[m] - c_prime_list[m] * O_list[m + 1]
            
        # 5. Build Final Tensor Output 
        O_flat = torch.stack(O_list, dim=0)

        O_flat = torch.where(no_data[None, :], torch.zeros((), dtype=O_flat.dtype, device=O_flat.device), O_flat)

        # Objects in the precision of the input
        return O_flat.reshape(M, Nx, Ny).to(S.dtype)

    def solve_object_time_evolution_batched(self, S, D, Reg_Space, lambda_s, lambda_t):
        """
        solve_object_time_evolution for all the sequences of a batch at once, with one time weight: the same
        operations element by element, with the sequences as a leading dimension, so the result is the same as
        solving each sequence in turn (the threshold of the frequencies without data is still taken per sequence).
        A time-series fit called the single-sequence solver once per patch and evaluation.

        Parameters:
        -----------
        S, D : torch.Tensor (complex)
            Shape (n_seq, M, J, Nx, Ny): the OTFs and the Fourier transformed frames.
        Reg_Space, lambda_s : as in solve_object_time_evolution
        lambda_t : float or torch.Tensor
            One weight for every link, or a real tensor (n_seq, M - 1) with the weight of each link (t, t+1) of each
            sequence (the SNR rule of the `time` regularization).

        Returns:
        --------
        O : torch.Tensor (complex), shape (n_seq, M, Nx, Ny)
        """
        n_s, M, J, Nx, Ny = S.shape
        B = Nx * Ny
        per_link = torch.is_tensor(lambda_t) and lambda_t.dim() == 2

        sum_S_sq_flat = torch.sum(torch.abs(S)**2, dim=2).reshape(n_s, M, B)
        RHS = torch.sum(torch.conj(S) * D, dim=2).reshape(n_s, M, B)
        reg_s_flat = (lambda_s * Reg_Space).flatten()

        if M == 1:
            return (RHS / (sum_S_sq_flat + reg_s_flat + 1e-10)).reshape(n_s, M, Nx, Ny)

        lt_max = float(lambda_t.max()) if per_link else float(lambda_t)
        w_freq = torch.sum(sum_S_sq_flat, dim=1)
        no_data = (w_freq <= 1e-6 * lt_max) | (w_freq <= 1e-10 * torch.max(w_freq, dim=1, keepdim=True).values)
        if torch.any(no_data):
            sum_S_sq_flat = torch.where(no_data[:, None, :], torch.ones_like(sum_S_sq_flat), sum_S_sq_flat)

        work = torch.complex64 if S.device.type == 'mps' else torch.complex128
        if per_link:
            lam = lambda_t.to(device=S.device, dtype=sum_S_sq_flat.dtype)[:, :, None]        # (n_s, M - 1, 1)
            a = (-lam).expand(n_s, M - 1, B).to(work)
            c = a
            d_rows = []
            for m in range(M):
                row = sum_S_sq_flat[:, m, :] + reg_s_flat
                if m > 0:
                    row = row + lam[:, m - 1]
                if m < M - 1:
                    row = row + lam[:, m]
                d_rows.append(row)
        else:
            a = torch.full((n_s, M - 1, B), -lambda_t, dtype=work, device=S.device)
            c = torch.full((n_s, M - 1, B), -lambda_t, dtype=work, device=S.device)
            d_rows = []
            for m in range(M):
                row = sum_S_sq_flat[:, m, :] + reg_s_flat
                if m == 0 or m == M - 1:
                    row = row + lambda_t
                else:
                    row = row + 2.0 * lambda_t
                d_rows.append(row)

        c_prime_list = [c[:, 0, :] / d_rows[0]]
        d_prime_list = [RHS[:, 0, :] / d_rows[0]]
        for m in range(1, M - 1):
            denom = d_rows[m] - a[:, m - 1, :] * c_prime_list[m - 1]
            c_prime_list.append(c[:, m, :] / denom)
            d_prime_list.append((RHS[:, m, :] - a[:, m - 1, :] * d_prime_list[m - 1]) / denom)
        denom = d_rows[M - 1] - a[:, M - 2, :] * c_prime_list[M - 2]
        d_prime_list.append((RHS[:, M - 1, :] - a[:, M - 2, :] * d_prime_list[M - 2]) / denom)

        O_list = [None] * M
        O_list[M - 1] = d_prime_list[M - 1]
        for m in range(M - 2, -1, -1):
            O_list[m] = d_prime_list[m] - c_prime_list[m] * O_list[m + 1]

        O_flat = torch.stack(O_list, dim=1)
        O_flat = torch.where(no_data[:, None, :], torch.zeros((), dtype=O_flat.dtype, device=O_flat.device), O_flat)
        return O_flat.reshape(n_s, M, Nx, Ny).to(S.dtype)

    def compute_object(self, images_ft, otf, sigma, plane, type_filter='tophat', pars_s0=None, pars_s2=None, need_filter=True):
        """
        Compute the object in Fourier space using the specified filter.
        Parameters:
        --------
        images_ft (torch.Tensor): 
            The Fourier transform of the observed images.
        otf (torch.Tensor): 
            The Fourier transform of the point spread function (PSF).
        type_filter (str, optional): 
            The type of filter to use ('tophat'/'scharmer'/'garrote'). Default is 'tophat'.
        need_filter (bool, optional):
            If False, the scharmer or garrote mask is not computed and the filtered object carries only the
            diffraction mask; the unfiltered object (the first output) is the same. For callers that use only
            the unfiltered object. Default is True.
        Returns:
        --------
        torch.Tensor: The computed object in Fourier space.
        """

        out_ft = [None] * self.n_o
        out_filter_ft = [None] * self.n_o
        out_filter = [None] * self.n_o
        
        for i in range(self.n_o):

            frames_mean = torch.mean(images_ft[i][:, :, 0, 0], dim=1).real

            s_u, s2, K, v0, p = self.get_su_s2(obj=i, sigma=sigma, pars_s0=pars_s0, pars_s2=pars_s2, frames_mean=frames_mean)
            
            # Per-frame noise weights γ_kj = s2 / σ_kj², dimensionless and referred to the variance s2 that enters
            # s2/s_u and the filters. The joint s2 is one variance for the whole batch, so the weights are referred
            # to it: each frame then counts as 1/σ² and each sequence or time step keeps its own noise, whatever
            # the other sequences of the batch are. The marginal s2 is fitted per sequence, so there the weights
            # stay relative to each sequence's mean σ².
            if self.loss_type == 'marginal':
                s2_local = torch.mean(sigma[i]**2, dim=1)
            else:
                s2_local = s2 * torch.ones_like(sigma[i][:, 0])
            gamma = s2_local[:, None, None, None] / (sigma[i][:, :, None, None] ** 2 + 1e-10)

            time_coupled = self.loss_filter == 'wiener_time' or self.loss_filter == 'wiener_time_contrast'

            # The loss of the time path (need_filter=False) uses only the unfiltered object of the time solve: the sums
            # below, the mask and the filtered object are skipped there. In a time-series fit they were computed in
            # every evaluation and thrown away (13 % of the fit of a window of the time example, the result bit for bit
            # the same, 2026-11-22)
            lean = time_coupled and not need_filter
            if not lean:
                Sconj_S = torch.sum(gamma * torch.conj(otf[i]) * otf[i], dim=1)
                Sconj_I = torch.sum(gamma * torch.conj(otf[i]) * images_ft[i], dim=1)

            if pars_s2 is not None:
                s2_term = s2
                s2_filt = s2
            else:
                s2_term = s2[:, None, None] if hasattr(s2, 'dim') and s2.dim() > 0 else s2
                s2_filt = s2[:, None] if hasattr(s2, 'dim') and s2.dim() > 0 else s2

            if time_coupled and self.enforce_positive_object:
                raise ValueError("enforce_positive_object is not available with the time-coupled filters: the positivity ADMM has no time coupling")

            if time_coupled:
                # Spatial regularisation of the time solve: none for scharmer and garrote (their gain selects
                # the frequencies), s2/s_u for tophat, so that lambda_t -> 0 gives back each filter's estimate
                plane_shape = dict(size=otf[i].shape[-2:], dtype=otf[i].real.dtype, device=otf[i].device)
                if self.image_filter[i] in ('scharmer', 'garrote'):
                    Reg_Space, lambda_s = torch.zeros(**plane_shape), 0.0
                else:
                    reg = s2_term / s_u
                    if torch.is_tensor(reg) and reg.dim() == 3:
                        reg = reg[0]
                    Reg_Space, lambda_s = reg * torch.ones(**plane_shape), 1.0

                # Per-frame weights gamma enter as sqrt(gamma) on both S and D: sum gamma|S|^2, sum gamma S* D
                # The weighted frames and the SNR weights of the links depend only on the frames and their noise,
                # which a fit does not change: computed in the first evaluation of a batch and kept (_time_constants,
                # emptied by _deconvolve_pass for every batch)
                constants = getattr(self, '_time_constants', None)
                cached = constants.get(i) if constants is not None and not sigma[i].requires_grad else None
                if cached is not None and cached[0] is images_ft[i] and cached[1] is sigma[i]:
                    w, images_tmp, lambda_t = cached[2:]
                    fresh = False
                else:
                    w = torch.sqrt(gamma)
                    images_tmp = rearrange(w * images_ft[i], '(s t) f w h -> s t f w h', t=self.n_t)
                    # lambda_t competes with sum_j |S_j|^2, which scales as n_frames / npix^2: the configured value is
                    # the weight for 64-px patches and 12 frames, and means the same for any patch size and frame count
                    lambda_t = self.time_regularization * (64.0 / self.npix)**2 * (images_tmp.shape[2] / 12.0)
                    fresh = True
                otf_tmp = rearrange(w * otf[i], '(s t) f w h -> s t f w h', t=self.n_t)
                images_raw = rearrange(images_ft[i], '(s t) f w h -> s t f w h', t=self.n_t)
                n_seq = images_tmp.shape[0]

                out = []
                if self.loss_filter == 'wiener_time' and fresh:
                    snr_threshold = float(getattr(self, "time_snr_threshold", SNR_RULE_DEFAULT_THRESHOLD))      # set by set_regularizations
                    if snr_threshold > 0 and self.n_t > 1:
                        # SNR rule: a link to a step whose frames are below the threshold is weighted up, so that step
                        # leans on its neighbours (and anchors its PSFs) instead of fitting a sharp PSF to its noise.
                        # The SNR comes from the data alone, the same in every evaluation of the fit
                        if getattr(self, 'time_snr_source', 'frames') == 'power':
                            # restored products have no noise band beyond the cutoff: a step is scored by its power at
                            # 0.1-0.3 of the cutoff against the median step of its series, which sits at the threshold
                            snr = snr_threshold * step_power_ratio(images_tmp.detach(), self.rho[i].to(images_tmp.device))
                        else:
                            snr = step_snr(images_tmp.detach(), self.rho[i].to(images_tmp.device), fourier=True)
                        lambda_t = lambda_t * snr_link_factors(snr.double(), snr_threshold, getattr(self, "time_snr_reference", "window"))
                if fresh and constants is not None and not sigma[i].requires_grad:
                    constants[i] = (images_ft[i], sigma[i], w, images_tmp, lambda_t)
                if self.loss_filter == 'wiener_time':
                    # All sequences at once: the same recursion per frequency, without a Python loop over the patches
                    out = [self.solve_object_time_evolution_batched(otf_tmp, images_tmp, Reg_Space, lambda_s, lambda_t)]
                for j in range(n_seq if self.loss_filter == 'wiener_time_contrast' else 0):
                    if self.loss_filter == 'wiener_time_contrast':
                        # rms of each step's frames without their mean. Dividing by the mean of the patch, as before,
                        # divided by the zero frequency of apodized patches whose mean remove_gradient_apodization has
                        # taken out: near 0 and of either sign, it gave "contrasts" of -200 to 30 and could turn the
                        # time weight negative. Only the ratios between steps are used, and the frames of a sequence
                        # share their mean, so the rms ranks the steps as the contrast would
                        tmp = torch.sum(torch.abs(images_raw[j, :, ...])**2, dim=(-1, -2)) - torch.abs(images_raw[j, :, :, 0, 0])**2
                        tmp = torch.mean(tmp, dim=-1)
                        contrast = torch.sqrt(tmp / self.npix**4).real
                    else:
                        contrast = None
                    
                    tmp = self.solve_object_time_evolution(otf_tmp[j, ...], images_tmp[j, ...], Reg_Space, lambda_s, lambda_t, contrast=contrast)
                    out.append(tmp[None, ...])
                out = torch.cat(out, dim=0)
                time_ft = rearrange(out, 's t w h -> (s t) w h')

            if lean:
                out_ft[i] = time_ft
                continue

            # Lofdahl & Scharmer (1994) mask, or its continuous gain
            if self.image_filter[i] in ('scharmer', 'garrote'):
                # In the time path the object of a step also carries what the time term brings from its neighbours,
                # so its mask is computed from the sums over the neighbouring steps: from the step alone, a noisy
                # step keeps only a few frequencies and loses what its neighbours provide
                SS_mask, SI_mask = Sconj_S, Sconj_I
                if time_coupled:
                    SS_mask, SI_mask = self.time_window_sums(Sconj_S, Sconj_I)
                if not need_filter:
                    # The caller only uses the unfiltered object (the loss of the time path): the mask, whose
                    # scharmer flood fill runs on the CPU, took 62 % of a time-series fit and was thrown away
                    mask = self.mask_diffraction_th[i][None, :, :]
                elif self.image_filter[i] == 'scharmer':
                    mask = self.lofdahl_scharmer_filter(SS_mask, SI_mask, s2_filt.detach()) * self.mask_diffraction_th[i][None, :, :]
                else:
                    mask = self.garrote_filter(SS_mask, SI_mask, s2_filt) * self.mask_diffraction_th[i][None, :, :]
                
                if time_coupled:
                    out_ft[i] = time_ft
                else:
                    out_ft[i] = Sconj_I / (Sconj_S + 1e-10)
                            
                out_filter_ft[i] = out_ft[i] * mask

                # If we want to enforce positivity of the object, we can use ADMM to solve the optimization problem
                if self.enforce_positive_object:
                    out_filter_ft[i] = self.enforce_positive_admm(Sconj_I, Sconj_S, mask, iters=500)
                
            # Use simple Wiener filter with tophat prior            
            if (self.image_filter[i] == 'tophat'):
                if time_coupled:
                    out_ft[i] = time_ft
                else:
                    out_ft[i] = Sconj_I / (Sconj_S + s2_term / s_u)
                
                out_filter_ft[i] = out_ft[i] * self.mask_diffraction_th[i][None, :, :]

            out_filter[i] = torch.fft.ifft2(out_filter_ft[i]).real
            
            # Even though we are enforcing non-negativity, FFT+IFFT can introduce small 
            # negative values due to numerical errors, so we clamp the output to ensure non-negativity.
            if self.enforce_positive_object:
                out_filter[i] = torch.clamp(out_filter[i], min=0.0)
            
            # Add the gradient that we removed
            if self.remove_gradient_apodization and plane is not None:
                 out_filter[i] += plane[i][:, 0, :, :]
        
        return out_ft, out_filter_ft, out_filter
    
    def enforce_positive_admm(self, Sconj_I, Sconj_S, mask, iters=10):
        """
        Enforce positive object using ADMM        

        

        Args:
            Sconj_I (_type_): _description_
            Sconj_S (_type_): _description_
            mask (_type_): _description_
            iters (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        O_ft = Sconj_I / (Sconj_S + 1e-10) * mask
        O = torch.fft.ifft2(O_ft).real

        U = torch.zeros_like(O).to(self.device)
        V = O.clone().to(self.device)
        V = torch.clamp(V, min=0.0)

        rho = 1e-4# * Sconj_S.mean().real
        
        t = tqdm(range(iters), desc="ADMM iterations")

        for _ in t:
            # Update O
            O_ft = (Sconj_I + rho * torch.fft.fft2(V - U)) / (Sconj_S + rho) * mask
            O = torch.fft.ifft2(O_ft)                    
            
            # Update V
            V = torch.clamp((O + U).real, min=0.0)

            # Update U
            U = U + O - V

            tmp = OrderedDict()
            tmp['O_min'] = torch.min(O.real).item()
            tmp['O_max'] = torch.max(O.real).item()
            tmp['V_min'] = torch.min(V).item()
            tmp['V_max'] = torch.max(V).item()

            t.set_postfix(ordered_dict=tmp)
                    
        return torch.fft.fft2(V)
    
    def compute_loss_half(self, frames_ft, otf_half, sigma):
        """
        The joint wiener loss of compute_loss computed on half of the spectrum (optimization.half_spectrum).

        The frames and the PSFs are real, so every term of the loss at frequency -k equals the one at k. The OTFs come
        from rfft2 (columns 0 .. n/2) and the sums run over that half: columns 1 .. n/2-1 stand for themselves and their
        mirror columns, so they count twice, column n/2 once; row 0 and column 0 are dropped as in compute_loss. The
        value and its gradient are those of compute_loss to float32 rounding (relative 1e-7 on the example burst with
        2, 10 and 30 modes). The FFT of the PSFs and the loss sums cost half: the fit of the example burst (65 patches,
        2 cameras, LBFGS 10) takes 4.75 s instead of 5.53 s on a V100S, 14.45 s instead of 15.5 s on a T4 (2026-11-24).
        Only the joint loss with the wiener filter_loss, without jitter. On by default (optimization.half_spectrum).
        """
        n = self.npix
        h = n // 2 + 1
        w = torch.full((h,), 2.0, device=self.device)
        w[0] = 1.0
        w[-1] = 1.0
        total = torch.tensor(0.0).to(self.device)
        for i in range(self.n_o):
            frames_mean = torch.mean(frames_ft[i][:, :, 0, 0], dim=1).real
            s_u, s2, K, v0, p = self.get_su_s2(obj=i, sigma=sigma, frames_mean=frames_mean)
            s2 = torch.mean(sigma[i]**2, dim=1)
            s2_term = s2[:, None, None]
            s_u = s_u[..., 0:h] if torch.is_tensor(s_u) and s_u.dim() >= 2 else s_u
            du = frames_ft[i][..., 0:h]
            S = otf_half[i]
            hu2 = s2_term + s_u * torch.sum((S * torch.conj(S)).real, dim=1)
            du2 = torch.sum((du * torch.conj(du)).real, dim=1)
            hu_du = torch.sum(du * torch.conj(S), dim=1)
            loss_data = 0.5 * (du2 - s_u * (hu_du * torch.conj(hu_du)).real / hu2) / s2_term
            loss_data = loss_data * self.mask_diffraction_th[i][None, :, 0:h]
            total = total + torch.sum(loss_data[:, 1:, 1:] * w[None, None, 1:]) / (loss_data.shape[0] * (n - 1)**2)
        return total, torch.tensor(0.0).to(self.device), total

    def compute_loss(self, frames_ft, otf, sigma, type_filter='tophat', pars_s0=None, pars_s2=None, jitter=None):
        """
        Compute the object in Fourier space using the specified filter.
        Parameters:
        --------
        images_ft (torch.Tensor): 
            The Fourier transform of the observed images.
        otf (torch.Tensor): 
            The Fourier transform of the point spread function (PSF).
        type_filter (str, optional): 
            The type of filter to use ('tophat'/'scharmer'/'garrote'). Default is 'tophat'.
        Returns:
        --------
        torch.Tensor: The computed object in Fourier space.
        """

        out_ft = [None] * self.n_o
        out_filter_ft = [None] * self.n_o
        out_filter = [None] * self.n_o

        loss_data_total = torch.tensor(0.0).to(self.device)
        loss_prior_total = torch.tensor(0.0).to(self.device)
        loss_total = torch.tensor(0.0).to(self.device)

        # Compute the mean of the frames in Fourier space (frequency (0,0) to use it in the s_u prior        
        
        # If we want to do time regularization, first get the object with the Wiener filter
        if self.loss_filter == 'wiener_time' or self.loss_filter == 'wiener_time_contrast':
            # The loss uses only the unfiltered object: no scharmer/garrote mask (need_filter=False)
            obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, 
                                                                        otf, 
                                                                        sigma, 
                                                                        None, 
                                                                        pars_s0=pars_s0, 
                                                                        pars_s2=pars_s2,
                                                                        need_filter=False)
        
        for i in range(self.n_o):            

            if self.loss_filter == 'wiener':

                frames_mean = torch.mean(frames_ft[i][:, :, 0, 0], dim=1).real

                s_u, s2, K, v0, p = self.get_su_s2(obj=i, sigma=sigma, pars_s0=pars_s0, pars_s2=pars_s2, frames_mean=frames_mean)

                # One noise variance per sequence, as in the lowpass loss: get_su_s2 gives the joint loss one
                # variance for the whole batch, which made the PSFs fitted to a sequence depend on the noise of
                # the other sequences of its batch
                if self.loss_type != 'marginal':
                    s2 = torch.mean(sigma[i]**2, dim=1)

                # The value of s_u in the case of the joint estimation should be
                # selected by hand to give good results
                # sigma**2/s_u should be the ration between noise and estimated object power spectrum
                # s2 = torch.mean(sigma[i]**2, dim=1) * 10
                
                # m_prior = torch.mean(frames_ft[i],dim=1,keepdims=True)
                # du = frames_ft[i] - m_prior * otf[i]
                
                du = frames_ft[i]
                if pars_s2 is not None:
                    s2_term = s2
                else:
                    s2_term = s2[:, None, None] if hasattr(s2, 'dim') and s2.dim() > 0 else s2

                hu2 = s2_term + s_u * torch.sum(otf[i] * torch.conj(otf[i]), dim=1)
                du2 = torch.sum(du * torch.conj(du), dim=1)
                hu_du = torch.sum(du * torch.conj(otf[i]), dim=1)
                hu_du2 = s_u * hu_du * torch.conj(hu_du)
                                
                loss_data = 0.5 * (du2 - hu_du2 / hu2) / s2_term

                loss_data *= self.mask_diffraction_th[i][None, :, :]
                
                # If we are doing a marginal estimation of the object, we 
                # # need to add the effect of the marginalized object and also add 
                # a prior on the parameters of s_u and on s2 to keep them in a reasonable range
                if self.loss_type == 'marginal':
                    if pars_s2 is not None:
                        self.pars_s0_avg = [K.detach(), v0.detach(), p.detach(), s2.detach()]
                        self.pars_s0_out[i, 0] = K.detach()
                        self.pars_s0_out[i, 1] = v0.detach()
                        self.pars_s0_out[i, 2] = p.detach()
                        self.pars_s2_out[i, 0] = s2.detach()
                        
                        # Prior loss consequence of the marginalization of the object in the joing loss
                        # plus the term depending on the noise variance
                        loss_prior_marginal = 0.5 * torch.log(hu2) + 0.5 * (frames_ft[i].shape[1] - 1.0) * torch.log(s2)
                        loss_prior = loss_prior_marginal
                    else:
                        self.pars_s0_avg = [torch.mean(K).detach(), torch.mean(v0).detach(), torch.mean(p).detach(), torch.mean(s2).detach()]
                        self.pars_s0_out[:, i, 0] = K[:, 0, 0].detach()
                        self.pars_s0_out[:, i, 1] = v0[:, 0, 0].detach()
                        self.pars_s0_out[:, i, 2] = p[:, 0, 0].detach()
                        self.pars_s0_out[:, i, 3] = s2.detach()
                        
                        # Prior loss consequence of the marginalization of the object in the joing loss
                        # plus the term depending on the noise variance
                        loss_prior_marginal = 0.5 * torch.log(hu2) + 0.5 * (frames_ft[i].shape[1] - 1.0) * torch.log(s2[:, None, None])

                        # Prior loss on sigma**2 to avoid zero division and to keep it in a reasonable range
                        # We use a Gaussian prior on log(sigma**2) with mean given by the average of sigma**2 
                        # and a sufficiently large variance to avoid constraining it too much
                        sig2 = torch.mean(sigma[i]**2, dim=1)[:, None, None]
                        loss_prior_s2 = (torch.log(s2[:, None, None]) - torch.log(sig2))**2

                        # Prior loss on K, v0 and p to keep them in a reasonable range

                        # Gaussian prior on log(K) with mean (peak power spectrum for normalized images) and variance 1.0
                        loss_prior_K = 0.5 * (torch.log10(K) - np.log10(self.K_prior[0]))**2 / self.K_prior[1]**2 + torch.log10(K)
                        
                        # Gaussian prior on log(v0) with mean log(0.1) (cutoff frequency for the power spectrum) and variance 1.0
                        loss_prior_v0 = 0.5 * (torch.log10(v0) - np.log10(self.v0_prior[0]))**2 / self.v0_prior[1]**2
                        
                        # Gaussian prior on log(p) with mean log(2.0) (power law index for the power spectrum) and variance 1.0
                        loss_prior_p = 0.5 * (p - self.p_prior[0])**2 / self.p_prior[1]**2

                        loss_prior = loss_prior_marginal + loss_prior_K + loss_prior_p + loss_prior_v0 + loss_prior_s2
                    
                    loss_prior *= self.mask_diffraction_th[i][None, :, :]

                    loss = loss_data + loss_prior
                else:
                    loss_prior = torch.tensor(0.0).to(self.device)
                    loss = loss_data

            if self.loss_filter == 'wiener_time' or self.loss_filter == 'wiener_time_contrast':

                frames_mean = torch.mean(frames_ft[i][:, :, 0, 0], dim=1).real

                s_u, s2, K, v0, p = self.get_su_s2(obj=i, sigma=sigma, pars_s0=pars_s0, pars_s2=pars_s2, frames_mean=frames_mean)

                # One noise variance per sequence and time step, as in the wiener loss: with the variance of the
                # whole batch a much noisier step weighed as much as a quiet one and spoilt the PSFs fitted to all
                if self.loss_type != 'marginal':
                    s2 = torch.mean(sigma[i]**2, dim=1)

                if pars_s2 is not None:
                    s2_term = s2
                else:
                    s2_term = s2[:, None, None] if hasattr(s2, 'dim') and s2.dim() > 0 else s2
                                
                residual = frames_ft[i] - obj_ft[i][:, None, ...] * otf[i]
                loss_data = 0.5 * torch.sum(residual * torch.conj(residual), dim=1) / s2_term
                
                loss_data *= self.mask_diffraction_th[i][None, :, :]
                                                
                loss_prior = torch.tensor(0.0).to(self.device)
                loss = loss_data
                    
            if self.loss_filter == 'lowpass':

                s_u, s2, K, v0, p = self.get_su_s2(obj=i, sigma=sigma)

                # One noise variance per sequence, (n_seq, 1, 1)
                s2_seq = torch.mean(sigma[i]**2, dim=1)[:, None, None]
                                
                du = frames_ft[i]
                Q = torch.sum(otf[i] * torch.conj(otf[i]), dim=1).real
                hu2 = s2_seq + s_u * Q
                du2 = torch.sum(du * torch.conj(du), dim=1)
                hu_du = torch.sum(du * torch.conj(otf[i]), dim=1)
                hu_du2 = hu_du * torch.conj(hu_du)

                loss_prior = torch.tensor(0.0).to(self.device)
                
                # Wiener data term weighted by the Lofdahl & Scharmer (1994) mask, or by its continuous gain
                if self.image_filter[i] in ('scharmer', 'garrote'):

                    # We assume, for the moment, that the noise is the same for all frames
                    Sconj_S = torch.sum(torch.conj(otf[i]) * otf[i], dim=1)
                    Sconj_I = torch.sum(torch.conj(otf[i]) * frames_ft[i], dim=1)

                    if self.image_filter[i] == 'scharmer':
                        mask = self.lofdahl_scharmer_filter(Sconj_S, Sconj_I, sigma[i]**2) * self.mask_diffraction_th[i][None, :, :]
                    else:
                        mask = self.garrote_filter(Sconj_S, Sconj_I, sigma[i]**2) * self.mask_diffraction_th[i][None, :, :]

                    loss_data = 0.5 * (du2 - s_u * hu_du2 / hu2) / s2_seq * mask
                
                # Gonsalves error metric (no object prior) inside the diffraction cutoff; the floor is
                # relative to the largest sum |S|^2 of each sequence, Q = 0 only beyond the cutoff
                if (self.image_filter[i] == 'tophat'):
                    Q_floor = 1e-6 * torch.amax(Q, dim=(-1, -2), keepdim=True)
                    loss_data = 0.5 * (du2 - self.mask_diffraction_th[i][None, :, :] * hu_du2 / (Q + Q_floor)) / s2_seq

                loss = loss_data

            
            # Weighted version of the loss computation
            # weight_mask = 1.0 + self.rho[i]**2            
            # loss_data_total += torch.mean(weight_mask * loss_data).real
            # loss_prior_total += torch.mean(loss_prior).real
            # loss_total += torch.mean(weight_mask * loss).real

            # Remove zero frequency                     
            loss_data_total += torch.mean(loss_data[:, 1:, 1:].real)
            
            if self.loss_type == 'marginal':
                loss_prior_total += torch.mean(loss_prior[:, 1:, 1:].real)
            loss_total += torch.mean(loss[:, 1:, 1:].real)

            if jitter is not None and self.use_jitter == 'slow':                
                loss_total += 0.01 * torch.mean(jitter[:, :, 0, :]**2) + 0.01 * torch.mean((jitter[:, :, 1, :] - jitter[:, :, 0, :])**2)
        
        return loss_data_total, loss_prior_total, loss_total

    
    def psfs_from_parameters(self, modes, diversity, n_active, jitter=None, shift=None):
        """
        PSFs and OTFs from the fitted parameters, built as the closure builds them: the first
        n_active modes, tip-tilt referenced to its mean over the frames (and zeroed if psf.remove_tt) for the
        wavefront models, VAE shifts referenced to frame 0.

        Parameters
        ----------
        modes : torch.Tensor
            (n_seq, n_f, n_modes) coefficients or latents.
        diversity : list of torch.Tensor
            Diversity per object, (n_seq, n_frames).
        n_active : int
            Number of active modes in the last optimisation step.
        jitter : torch.Tensor, optional
        shift : tuple of torch.Tensor, optional
            (shift_x, shift_y) of the VAE, (n_seq, n_f).
        """
        if self.psf_model.lower() in ['zernike', 'kl']:
            modes_centered = modes.clone()
            modes_centered[:, :, 0:2] = modes_centered[:, :, 0:2] - modes[:, :, 0:2].mean(dim=1, keepdim=True)
            if self.remove_tt:
                modes_centered[:, :, 0:2] = 0.0
            return self.compute_psfs(modes_centered[..., 0:n_active], diversity, jitter=jitter)

        if self.psf_model.lower() == 'nmf':
            return self.compute_psfs_nmf(modes[..., 0:n_active])

        if self.psf_model.lower() == 'vae':
            if shift is None:
                shift = (torch.zeros(modes.shape[0:2], device=modes.device), torch.zeros(modes.shape[0:2], device=modes.device))
            sx = shift[0] - shift[0][:, 0:1]
            sy = shift[1] - shift[1][:, 0:1]
            return self.compute_psfs_vae(modes[..., 0:n_active], shift_x=sx, shift_y=sy)

    def object_from_parameters(self, obj, plane=None):
        """
        Object, its spectrum and the filtered object from the optimised object parameters
        (infer_object=True path).

        The spectra follow the convention of compute_object: obj_ft is the unnormalised DFT
        of the object, so that obj_ft * otf (otf with norm='ortho') is directly comparable
        with the ortho spectra of the frames, and torch.fft.ifft2 (default norm) returns
        the object in the units of the frames.

        Parameters
        ----------
        obj : list of torch.Tensor
            Object parameters (n_seq, nx, ny) for each object.
        plane : list of torch.Tensor, optional
            Gradient removed during apodization; added back to the filtered object when
            images.remove_gradient_apodization is set.
        """
        obj_ft = [None] * self.n_o
        obj_filter = [None] * self.n_o

        for i in range(self.n_o):
            if (self.config['optimization']['transform'] == 'softplus'):
                scale = self.config['optimization']['softplus_scale']
                tmp = F.softplus(obj[i] * scale) / scale
            else:
                tmp = torch.clamp(obj[i], min=0.0)
            obj_ft[i] = torch.fft.fft2(tmp)

        obj_filter_ft = self.fft_filter(obj_ft)

        for i in range(self.n_o):
            obj_filter[i] = torch.fft.ifft2(obj_filter_ft[i]).real
            if self.remove_gradient_apodization and plane is not None:
                obj_filter[i] = obj_filter[i] + plane[i][:, 0, :, :]

        return obj_ft, obj_filter_ft, obj_filter

    def fft_filter(self, image_ft):
        """
        Applies a Fourier filter to the input image in the frequency domain.

        Parameters:
        -----------
        image_ft : torch.Tensor
            The input image in the frequency domain (Fourier transformed).

        Returns:
        --------
        torch.Tensor
            The filtered image in the frequency domain.
        """
        out = [None] * self.n_o
        for i in range(self.n_o):
            out[i] = image_ft[i] * self.mask_diffraction_th[i][None, :, :]

        return out
            
    def add_frames(self, frames, sigma=None, id_object=0, id_diversity=0, diversity=0.0, XY=None):
        """
        Add frames to the deconvolution object.
        Parameters:
        -----------
        
        frames : torch.Tensor
            The input frames to be deconvolved (n_sequences, n_frames, nx, ny).
        sigma : torch.Tensor
            The noise standard deviation for each object.
        id_object : int, optional
            The object index to which the frames belong (default is 0).
        diversity : torch.Tensor, optional
            The diversity coefficient to use for the deconvolution (n_sequences, n_objects).
            If None, the diversity coefficient is set to zero for all objects.
        Returns:
        --------
        None
        """
        
        self.logger.info(f"Adding frames for object {id_object} - diversity {id_diversity} - defocus {diversity}")

        # If time is included, flatten the frames for sequences and time
        if frames.ndim == 5:                
            self.time_present = True
            self.n_seq, self.n_t, self.n_f, self.nx, self.ny = frames.shape
            frames = rearrange(frames, 's t f h w -> (s t) f h w', t=self.n_t, s=self.n_seq)
            # frames = frames.reshape((-1, self.n_f, self.nx, self.ny))
        else:
            self.time_present = False
            self.n_seq, self.n_f, self.nx, self.ny = frames.shape
            self.n_t = 1

        if self.nx != self.npix or self.ny != self.npix:
            raise ValueError(f"Frames are {self.nx}x{self.ny} pixels but images.n_pixel is {self.npix}")

        # A pattern fixed on the detector, found from the frames of each burst (all time steps of a sequence
        # together: the pattern does not change in time)
        if self.config['images']['remove_fixed_pattern']:
            n_burst = self.n_t * self.n_f
            if n_burst < 12:
                msg = f"Fixed-pattern removal skipped for object {id_object}: {n_burst} frames per burst, at least 12 are needed"
                self.logger.warning(msg)
                warnings.warn(msg, RuntimeWarning)
                self.fixed_pattern.append(None)
            else:
                wavelength = self.config[f'object{id_object+1}']['wavelength']
                diffraction_limit = wavelength * 1e-8 / self.config['telescope']['diameter'] * 206265.0
                cutoff = self.config['images']['pix_size'] / diffraction_limit
                burst = rearrange(frames, '(s t) f h w -> s (t f) h w', t=self.n_t, s=self.n_seq)
                burst, pattern, n_removed = util.remove_fixed_pattern(torch.as_tensor(burst), cutoff)
                frames = rearrange(burst, 's (t f) h w -> (s t) f h w', t=self.n_t, s=self.n_seq)
                self.fixed_pattern.append(pattern)
                self.logger.info(f"Fixed pattern removed: median {float(torch.median(n_removed)):.1f} frequencies inside the diffraction limit per burst, "
                                 f"pattern rms {float(torch.std(pattern)):.3e} (frames rms {float(torch.std(burst)):.3e})")
        else:
            self.fixed_pattern.append(None)
                
        if sigma is None:
            self.logger.info(f"Estimating noise...")            

            sigma = noise.compute_noise(frames).to(self.device)
            self.logger.info(f"   * Average noise: {torch.mean(sigma)}")            
        else:
            sigma = torch.as_tensor(sigma, dtype=torch.float32)
            if self.time_present and sigma.ndim == 3:
                sigma = rearrange(sigma, 's t f -> (s t) f', t=self.n_t, s=self.n_seq)
            sigma = sigma.to(self.device)

        self.ind_object.append(id_object)        
        self.ind_diversity.append(id_diversity)

        self.frames.append(frames)
        self.sigma.append(sigma)        

        # If diversity is a scalar, we need to create a tensor of the same size as the number
        # of sequences
        if np.isscalar(diversity):
            diversity = torch.full(frames.shape[0:1], diversity, dtype=torch.float32).to(self.device)
                
        self.diversity.append(diversity)

        if XY is not None:
            if not torch.is_tensor(XY):
                # stored as a tensor: the metapupil projection reads self.XY.cpu()
                self.XY = torch.tensor(np.asarray(XY, dtype='float32'))
            else:
                self.XY = XY

    def remove_frames(self):
        """
        Add frames to the deconvolution object.
        Parameters:
        -----------
        frames : torch.Tensor
            The input frames to be deconvolved (n_sequences, n_objects, n_frames, nx, ny).
        sigma : torch.Tensor
            The noise standard deviation for each object.
        id_object : int, optional
            The object index to which the frames belong (default is 0).
        diversity : torch.Tensor, optional
            The diversity coefficient to use for the deconvolution (n_sequences, n_objects).
            If None, the diversity coefficient is set to zero for all objects.
        Returns:
        --------
        None
        """
        
        self.logger.info(f"Removing frames for all objects...")

        
        self.ind_object = []
        self.ind_diversity = []

        self.frames = []
        self.sigma = []
        self.fixed_pattern = []
        self.diversity = []


    def combine_frames(self):
        """
        Combine the frames from all objects and sequences into a single tensor.
        Observations with different diversity channels are concatenated along the frame axis.
        Returns:
        --------
        torch.Tensor: A tensor of shape (n_sequences, n_objects, n_frames, nx, ny) containing the combined frames.
        """

        self.logger.info(f"Setting up frames...")

        # Get number of objects and number of diversity channels from the added frames
        self.n_bursts = len(self.ind_object)
        self.n_o = max(self.ind_object) + 1

        n_seq, n_f, n_x, n_y = self.frames[0].shape
        
        frames = [None] * self.n_o
        plane = [None] * self.n_o
        diversity = [None] * self.n_o
        sigma = [None] * self.n_o
        index_frames_diversity = [None] * self.n_o

        # Count the number of frames per object, taking into account the diversity channels
        n_frames_per_object = [0] * self.n_o
        n_diversity_per_object = [0] * self.n_o
        for i in range(self.n_bursts):
            n_frames_per_object[self.ind_object[i]] += n_f
            n_diversity_per_object[self.ind_object[i]] += 1
        self.n_frames_per_object = n_frames_per_object
        
        for i in range(self.n_o):
            frames[i] = torch.zeros(n_seq, n_frames_per_object[i], n_x, n_y)
            plane[i] = torch.zeros(n_seq, 1, n_x, n_y)
            diversity[i] = torch.zeros(n_seq, n_frames_per_object[i])
            sigma[i] = torch.zeros(n_seq, n_frames_per_object[i])
            index_frames_diversity[i] = [0] * n_diversity_per_object[i]

        sigma_max = 0.0
        for i in range(self.n_bursts):
            sigma_max = max(sigma_max, torch.max(self.sigma[i]))            

        
        for i in range(self.n_bursts):

            i_obj = self.ind_object[i]
            i_div = self.ind_diversity[i]

            f0 = i_div * n_f
            f1 = (i_div + 1) * n_f

            index_frames_diversity[i_obj][i_div] = f0

            frames[i_obj][:, f0:f1, :, :], subtract = util.apodize(self.frames[i], self.window, gradient=self.remove_gradient_apodization)
            if self.remove_gradient_apodization:
                plane[i_obj][:, :, :, :] = subtract
            
            # Set the diversity for the current object for all frames and for the sequence            
            diversity[i_obj][:, f0:f1] = self.diversity[i][:, None].expand(-1, n_f)
            
            sigma[i_obj][:, f0:f1] = self.sigma[i] #/ sigma_max
                    
        return frames, diversity, index_frames_diversity, sigma, plane
    
    def update_object(self, cutoffs=None):
        """
        Update the object estimate with new cutoffs in the Fourier filter.

        Parameters
        ----------
        cutoffs : list
            A list containing the new cutoffs for each object. Each cutoff contains two numbers, indicating the
            lower and upper frequencies for the transition.
        """

        if self.simultaneous_sequences is None:
            self.logger.info(f"Deconvolution has not been carried out yet")
            return
        
        if cutoffs is None:
            self.logger.info(f"No cutoffs provided")
            return
        
        # Recompute the diffraction masks with the new cutoffs
        self.logger.info('Recalculating object with new cutoffs in the Fourier filter...')

        for i in range(self.n_o):
            self.cutoff[i] = cutoffs[i]
            self.logger.info(f"     - Filter: {self.image_filter[i]} - cutoff: {self.cutoff[i]}...")            
            if self.loss_type == "joint":
                self.logger.info(f"     - s_u: {self.s_u[i]}")
        
        self.compute_diffraction_masks()

        n_seq, _, _, _ = self.frames_apodized[0].shape

        # If time is present, we flatten the sequence and time dimensions to have a single sequence dimension
        # but compute the indices of the new flattened sequences to be able to recover the original shape of 
        # the object at the end of the optimization        
        n_seq = self.n_seq * self.n_t
        ind = np.arange(n_seq)
        n_seq_total = n_seq
                        
        ind = self.split_and_merge(n_seq, self.n_t, self.simultaneous_sequences)

        # # Split sequences in batches
        # ind = np.arange(n_seq)

        # n_seq_total = n_seq

        # # Split the sequences in groups of simultaneous sequences to be computed in parallel
        # ind = np.array_split(ind, np.ceil(n_seq / self.simultaneous_sequences))

        n_sequences = len(ind)
        
        self.psf_seq = [None] * n_sequences        
        self.degraded_seq = [None] * n_sequences
        self.obj_seq = [None] * n_sequences
        self.obj_diffraction_seq = [None] * n_sequences
        
        for i_seq, seq in enumerate(ind):

            # Restriction of the wavefront of this batch, if a first pass computed one
            self.nullspace_P_batch = self.nullspace_P_seq[i_seq].to(self.device) if self.nullspace_P_seq is not None else None
                            
            if len(seq) > 1:
                self.logger.info(f"Processing sequences [{seq[0]+1},{seq[-1]+1}]/{n_seq_total}")
            else:
                self.logger.info(f"Processing sequence {seq[0]+1}/{n_seq_total}")

            frames_apodized_seq = []
            plane_seq = []
            frames_ft = []
            sigma_seq = []
            diversity_seq = []
            for i in range(self.n_o):
                frames_apodized_seq.append(self.frames_apodized[i][seq, ...].to(self.device))
                plane_seq.append(self.plane[i][seq, ...].to(self.device))
                frames_ft.append(self.frames_ft[i][seq, ...].to(self.device))
                sigma_seq.append(self.sigma[i][seq, ...].to(self.device))
                diversity_seq.append(self.diversity[i][seq, ...].to(self.device))
                
            n_seq = len(seq)

            if self.psf_model.lower() == 'nmf':
                sx, sy = self.shift_seq[i_seq] if self.shift_seq[i_seq] is not None else (None, None)
                psf, otf = self.compute_psfs_nmf(self.modes_seq[i_seq][..., 0:self.n_active_seq[i_seq]], shift_x=sx, shift_y=sy)
            else:
                psf, otf = self.psfs_from_parameters(self.modes_seq[i_seq], diversity_seq, self.n_active_seq[i_seq],
                                                     jitter=self.jitter_seq[i_seq] if self.use_jitter != 'none' else None,
                                                     shift=self.shift_seq[i_seq])
            
            if (self.infer_object):

                # Filtered object from the stored object parameters, with the new mask
                obj_ft, obj_filter_ft, obj_filter = self.object_from_parameters(self.obj_infer_seq[i_seq], plane_seq)

            else:
                pars_s0_arg, pars_s2_arg = self.pars_raw_seq[i_seq]
                obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, otf, sigma_seq, plane_seq, pars_s0=pars_s0_arg, pars_s2=pars_s2_arg)  
                                   

            obj_filter_diffraction = [None] * self.n_o
            degraded = [None] * self.n_o
            for i in range(self.n_o):                
                # Diffraction-limited object only for the wavefront models, as in deconvolve
                if self.psf_model.lower() in ['zernike', 'kl']:
                    obj_filter_diffraction[i] = torch.fft.ifft2(obj_filter_ft[i] * self.psf_diffraction_ft[i][None, :, :], norm=self.fft_norm).real.detach()
            
                # Compute final degraded images
                degraded_ft = obj_filter_ft[i][:, None, :, :] * otf[i]
                # obj_filter_ft is the unnormalised DFT of the object and otf the ortho OTF, so the product
                # is the ortho spectrum of the degraded image, so it is inverted with the same norm
                degraded[i] = torch.fft.ifft2(degraded_ft, norm=self.fft_norm).real
                        
            for i in range(self.n_o):
                psf[i] = psf[i].detach().cpu()
                degraded[i] = degraded[i].detach().cpu()
                obj_filter[i] = obj_filter[i].detach()

            self.psf_seq[i_seq] = psf
            self.degraded_seq[i_seq] = degraded
            self.obj_seq[i_seq] = obj_filter
            self.obj_diffraction_seq[i_seq] = obj_filter_diffraction

            tfinal = time.time()

            # del psf, degraded, obj_filter, obj_filter_diffraction, degraded_ft, obj_ft, obj_filter_ft, otf

        self.nullspace_P_batch = None

        # Concatenate the results from all sequences and all objects independently
        # self.psf = [None] * self.n_o
        # self.degraded = [None] * self.n_o
        self.obj = [None] * self.n_o
        self.obj_diffraction = [None] * self.n_o

        # for i in range(self.n_o):
        self.modes = torch.cat(self.modes_seq, dim=0)
                
        for i in range(self.n_o):
            # tmp = [self.psf_seq[j][i] for j in range(n_sequences)]
            # self.psf[i] = torch.cat(tmp, dim=0)

            # tmp = [self.degraded_seq[j][i] for j in range(n_sequences)]
            # self.degraded[i] = torch.cat(tmp, dim=0)

            tmp = [self.obj_seq[j][i] for j in range(n_sequences)]
            self.obj[i] = torch.cat(tmp, dim=0)
            if self.time_present:
                self.obj[i] = rearrange(self.obj[i], '(s t) h w -> s t h w', t=self.n_t, s=self.n_seq)

            if self.psf_model.lower() in ['zernike', 'kl']:
                tmp = [self.obj_diffraction_seq[j][i] for j in range(n_sequences)]
                self.obj_diffraction[i] = torch.cat(tmp, dim=0)
                if self.time_present:
                    self.obj_diffraction[i] = rearrange(self.obj_diffraction[i], '(s t) h w -> s t h w', t=self.n_t, s=self.n_seq)
        
        return 
    
    def write(self, filename, extra=None):
        """
        Write the deconvolved object to a file.
        Parameters:
        -----------
        filename : str
            The name of the file to which the object will be written.
        Returns:
        --------
        None
        """
        
        self.logger.info(f"Writing object to {filename}...")
        
        hdu = [fits.PrimaryHDU(self.modes.cpu().numpy())]
        for i in range(self.n_o):
            tmp = fits.ImageHDU(self.obj[i].cpu().numpy())
            tmp.header['OBJECT'] = f'Object {i+1}'
            if extra is not None:
                for k, v in extra.items():
                    tmp.header[k] = v
            hdu.append(tmp)
        
        hdu = fits.HDUList(hdu)
        hdu.writeto(filename, overwrite=True)

        return

    def split_and_merge(self, M, N, K):
        """
        Split a sequence of length M into chunks of size N, ensuring that the last chunk does not exceed K.
        Parameters:
        -----------
        M : int
            The length of the sequence to be split.
        N : int
            The size of each chunk.
        K : int
            The maximum allowed size for the last chunk.
        Returns:
        --------
        list of lists
            A list containing the split chunks of the sequence.
        """
        if N > K:
            raise ValueError(f"simultaneous_sequences = {K} is smaller than the number of time steps n_t = {N}: a batch must hold whole time series")
            
        sequence = list(range(M))
        # Calculate the largest multiple of N that is <= K
        step_size = (K // N) * N
        
        # Directly slice the main sequence
        return [sequence[i:i + step_size] for i in range(0, len(sequence), step_size)]
            
    def deconvolve(self,
                   simultaneous_sequences=1,
                   infer_object=False,
                   optimizer='adam',
                   obj_in=None,
                   modes_in=None,
                   n_iterations=20,
                   sequence_length=None):
        """
        Perform deconvolution on a set of frames (see _deconvolve_pass for the parameters).

        With psf.nullspace_d_r0 in the configuration the fit is done twice: the first pass is the usual one; from
        it nullspace_projectors finds, for each sequence, the combinations of wavefront coefficients that the data
        determine better than a Kolmogorov prior, and the second pass, from the same start, lets the wavefront
        change only along those. The results are those of the second pass; self.modes are the free coefficients,
        and the PSFs are built from their projection.
        """
        kwargs = dict(simultaneous_sequences=simultaneous_sequences, infer_object=infer_object, optimizer=optimizer,
                      obj_in=obj_in, modes_in=modes_in, n_iterations=n_iterations, sequence_length=sequence_length)

        # A pass replaces the per-burst diversity and noise by their combination over objects: keep the inputs
        inputs = (list(self.diversity), list(self.sigma))

        self.nullspace_P_seq = None
        # Refuse an unsupported restriction before the first pass, not after it (the full check is in nullspace_projectors)
        if self.nullspace_d_r0 is not None and (self.loss_type != 'joint' or infer_object or self.psf_model.lower() != 'kl'):
            raise ValueError("The wavefront restriction is implemented for the KL model and the joint loss, without infer_object")
        self._deconvolve_pass(**kwargs)

        if self.nullspace_d_r0 is not None:
            self.nullspace_P_seq = self.nullspace_projectors(self.nullspace_d_r0, self.nullspace_threshold)
            self.logger.info(f"Second pass with the wavefront restricted to the directions the data determine")
            self.diversity, self.sigma = list(inputs[0]), list(inputs[1])
            self._deconvolve_pass(**kwargs)

    def restrict_modes(self, modes):
        """
        Project the wavefront coefficients of the current batch, (n_seq, n_f, n_active), on the directions kept by
        nullspace_projectors. The modes not yet active in the annealing enter as zero.
        """
        n_seq, n_f, n_active = modes.shape
        full = torch.zeros((n_seq, n_f, self.n_modes), device=modes.device, dtype=modes.dtype)
        full[..., 0:n_active] = modes
        full = torch.einsum('spq,sq->sp', self.nullspace_P_batch, full.reshape(n_seq, -1))
        return full.reshape(n_seq, n_f, self.n_modes)[..., 0:n_active]

    def nullspace_projectors(self, d_r0, threshold=1.0):
        """
        For each batch of the last pass, the projector of every sequence on the wavefront directions that its data
        determine better than a Kolmogorov prior.

        With the object as unknown, the information that the data d_k = O_k t_k + n carry on the wavefront
        coefficients theta is F = (2/s2) sum_k |O_k|^2 Re[J_k^H (I - t_k t_k^H/|t_k|^2) J_k], over the half plane
        inside the diffraction cutoff, with t_k the OTFs of the frames at frequency k and J_k = dt_k/dtheta. It is
        evaluated at the fitted wavefront, with the fitted object and the noise of the frames. Whitened by the prior
        standard deviations S (KL variances of the basis times (D/r0)^(5/3), none on tip-tilt), the eigenvectors U_r
        of S F S with eigenvalue above `threshold` are the directions the data determine; P = S U_r U_r^T S^-1.

        Parameters
        ----------
        d_r0 : float
            D/r0 of the Kolmogorov prior.
        threshold : float
            Eigenvalue of S F S above which a direction is kept (1: the data and the prior weigh the same).

        Returns
        -------
        list of torch.Tensor
            One (n_seq, n_f n_modes, n_f n_modes) tensor per batch, on the CPU.
        """
        if self.psf_model.lower() != 'kl' or self.kl_variance is None or self.add_piston:
            raise ValueError("The wavefront restriction needs the KL model (without piston) and the KL variances of its basis file")
        if self.loss_type != 'joint' or self.infer_object or self.n_o != 1 or any(len(d) > 1 for d in self.init_frame_diversity):
            raise ValueError("The wavefront restriction is implemented for the joint loss, one object and no phase diversity")

        n_f, nm = self.n_f, self.n_modes
        sig = np.sqrt(np.asarray(self.kl_variance[0:nm], dtype='float64') * d_r0**(5.0 / 3.0))
        sig[0:2] = 10.0
        S = torch.tensor(np.tile(sig, n_f), dtype=torch.float64)

        basis = self.basis[0].double()
        pupil = self.pupil[0].double()
        def otf_of(theta):
            wf = torch.einsum('...m,mxy->...xy', theta, basis)
            ft = torch.fft.fft2(pupil * torch.exp(1j * wf), norm='ortho')
            psf = (ft.conj() * ft).real
            return torch.fft.fft2(psf / psf.sum((-1, -2), keepdim=True), norm='ortho')

        # Independent frequencies: half plane inside the cutoff, without k = 0
        k = np.fft.fftfreq(self.npix) * self.npix
        kx, ky = np.meshgrid(k, k, indexing='ij')
        half = torch.tensor((self.rho[0].cpu().numpy() <= 1.0) & ((kx > 0) | ((kx == 0) & (ky > 0))), device=self.device)

        # noise variance of each sequence (mean over its frames), as the losses use it
        s2_seq = torch.mean(self.sigma[0]**2, dim=-1).double()
        ind = self.split_and_merge(self.n_seq * self.n_t, self.n_t, self.simultaneous_sequences)
        P_seq, kept = [], []
        for i_seq, seq in enumerate(ind):
            modes = self.modes_seq[i_seq].double().to(self.device)
            modes[:, :, 0:2] = modes[:, :, 0:2] - modes[:, :, 0:2].mean(dim=1, keepdim=True)
            otf = torch.fft.fft2(self.psf_seq[i_seq][0].to(self.device), norm='ortho')
            with torch.no_grad():
                _, obj_ft, _ = self.compute_object([self.frames_ft[0][seq, ...].to(self.device)], [otf],
                                                   [self.sigma[0][seq, ...].to(self.device)], [self.plane[0][seq, ...].to(self.device)])
            O2 = (obj_ft[0].abs()**2).double()
            P = torch.zeros((len(seq), n_f * nm, n_f * nm))
            for j in range(len(seq)):
                Jr = torch.func.vmap(torch.func.jacfwd(lambda t: torch.view_as_real(otf_of(t))))(modes[j])
                G = torch.complex(Jr[..., 0, :], Jr[..., 1, :])[:, half, :].permute(1, 0, 2)
                with torch.no_grad():
                    t = otf_of(modes[j])[:, half].T
                w = (2.0 / float(s2_seq[seq[j]])) * O2[j][half]
                F = torch.zeros((n_f * nm, n_f * nm), dtype=torch.float64, device=self.device)
                for f in range(n_f):
                    F[f * nm:(f + 1) * nm, f * nm:(f + 1) * nm] = ((G[:, f, :].conj().T * w) @ G[:, f, :]).real
                V = (G.conj() * t[:, :, None]).reshape(G.shape[0], n_f * nm)
                F = F - ((V.conj().T * (w / (t.abs()**2).sum(1))) @ V).real
                F = 0.5 * (F + F.T).cpu()
                ev, U = torch.linalg.eigh(S[:, None] * F * S[None, :])
                U = U[:, ev > threshold]
                P[j] = (S[:, None] * (U @ U.T) / S[None, :]).float()
                kept.append(int((ev > threshold).sum()))
            P_seq.append(P)

        self.nullspace_kept = kept
        self.logger.info(f"Wavefront directions kept per sequence (D/r0 {d_r0}, threshold {threshold}): median {np.median(kept):.0f} of {n_f * nm}")
        return P_seq

    def _deconvolve_pass(self,                                    
                   simultaneous_sequences=1,
                   infer_object=False,
                   optimizer='adam',
                   obj_in=None,
                   modes_in=None,
                   n_iterations=20,
                   sequence_length=None):
        

        """
        Perform deconvolution on a set of frames using specified parameters.
        Parameters:
        -----------
        frames : torch.Tensor
            List with the input frames to be deconvolved (n_batch, n_sequences, n_frames, nx, ny) for each object.
        sigma : torch.Tensor
            The noise standard deviation for each object.
        simultaneous_sequences : int, optional
            Number of sequences to be processed simultaneously (default is 1).
        infer_object : bool, optional
            Whether to infer the object during optimization (default is False).
        optimizer : str, optional
            The optimizer to use ('adam' for Adam, 'lbfgs' for LBFGS) (default is 'adam').
        obj_in : torch.Tensor, optional
            Initial object to use for deconvolution (default is None).
        modes_in : torch.Tensor, optional
            Initial modes to use for deconvolution (default is None).
        annealing : bool or str, optional
            Annealing schedule to use ('linear', 'sigmoid', 'none') (default is 'linear'').
        n_iterations : int, optional
            Number of iterations for the optimization (default is 20).        
        Returns:
        --------
        None
        """
                
        # Estimate the modes                
        # modes = self.modalnet(frames)

        self.simultaneous_sequences = simultaneous_sequences
        self.infer_object = infer_object

        _, self.n_f, self.n_x, self.n_y = self.frames[0].shape

        self.logger.info(f" *****************************************")
        self.logger.info(f" *** SPATIALLY INVARIANT DECONVOLUTION ***")
        self.logger.info(f" *****************************************")
        
        # Combine all frames        
        self.frames_apodized, self.diversity, self.init_frame_diversity, self.sigma, self.plane = self.combine_frames()
        
        # Precompute the FFT of the frames to speed up the optimization
        self.frames_ft = [None] * self.n_o
        for i in range(self.n_o):
            self.frames_ft[i] = torch.fft.fft2(self.frames_apodized[i], norm=self.fft_norm)

        # Define all basis
        self._define_basis()

        # Phase diversity acts on the wavefront; the nmf and vae PSFs have none
        if self.psf_model.lower() in ['nmf', 'vae']:
            for i in range(self.n_o):
                if len(self.init_frame_diversity[i]) > 1:
                    raise ValueError(f"Phase diversity ({len(self.init_frame_diversity[i])} channels for object {i}) needs a wavefront PSF model (zernike or kl), not {self.psf_model}")
        
        # Fill the list of frames and apodize the frames if needed
        # for i in range(self.n_o):
        #     self.frames_apodized[i] = self.frames_apodized[i].to(self.device)
        #     self.diversity[i] = self.diversity[i].to(self.device)
        #     self.sigma[i] = self.sigma[i].to(self.device)
                    
        self.logger.info(f"Frames")        
        for i in range(self.n_o):
            
            self.logger.info(f"  * Object {i}")            
            self.logger.info(f"     - Number of sequences {self.n_seq}...")
            if self.time_present:
                self.logger.info(f"     - Number of times {self.n_t}...")
            self.logger.info(f"     - Number of frames {self.n_f}...")
            self.logger.info(f"     - Number of diversity channels {len(self.init_frame_diversity[i])}...")
            for j, ind in enumerate(self.init_frame_diversity[i]):
                self.logger.info(f"       -> Diversity {j} = {self.diversity[i][0, ind]} - Noise = {self.sigma[i][0, ind]}...")
            self.logger.info(f"     - Size of frames {self.n_x} x {self.n_y}...")
            self.logger.info(f"     - Filter: {self.image_filter[i]} - cutoff: {self.cutoff[i]}...")
            if self.loss_type == "joint":
                self.logger.info(f"     - s_u: {self.s_u[i] * self.frames_prior_scale(i)} (s_u_joint for 64 px and at most 12 frames: {self.config[f'object{i+1}']['s_u_joint']})")
            if self.loss_type == "marginal":                
                self.logger.info(f"     - Stopping optimization after {self.stop_psd} modes")
                                
        self.finite_difference = util.FiniteDifference().to(self.device)
        self.set_regularizations()
                                                    
        # Compute the diffraction masks
        self.compute_diffraction_masks()
        
        # Annealing schedules

        if self.psf_model.lower() in ['zernike', 'kl']:
            modes = np.cumsum(np.arange(2, self.noll_max+1))

        if self.psf_model.lower() == 'nmf':
            n = max(2, (self.n_modes - 2) // 5)
            modes = np.linspace(2, self.n_modes, n).astype(int)

        if self.psf_model.lower() == 'vae':
            n = max(2, (self.n_modes - 2) // 5)
            modes = np.linspace(2, self.n_modes, n).astype(int)
            # modes = (self.n_modes * np.ones(n)).astype(int)
        
        self.anneal = self.compute_annealing(modes, n_iterations)
                
        # If the regularization parameter is a scalar, we assume that it is the same for all objects
        for reg in self.regularization:
            if reg.type == 'iuwt':                
                if not isinstance(reg.lambda_reg, list):
                    reg.lambda_reg = [reg.lambda_reg] * self.n_o

        # Initial values for the 
        if self.loss_type == 'marginal':
            if 'psd' in self.config:
                self.K_prior = [self.config['psd']['K'], 1.0]
                self.v0_prior = [self.config['psd']['v0'], 1.0]
                self.p_prior = [self.config['psd']['p'], 2.0]
                self.logger.info(f"PSD initial values :")
                self.logger.info(f"  - K: {self.K_prior[0]}")
                self.logger.info(f"  - v0: {self.v0_prior[0]}") 
                self.logger.info(f"  - p: {self.p_prior[0]}")
            else:
                self.K_prior = [self.config['priors']['K']['mean'], self.config['priors']['K']['sigma']]
                self.v0_prior = [self.config['priors']['v0']['mean'], self.config['priors']['v0']['sigma']]
                self.p_prior = [self.config['priors']['p']['mean'], self.config['priors']['p']['sigma']]
                self.logger.info(f"Normal hyperpriors parameters :")
                self.logger.info(f"  - K: mean = {self.K_prior[0]}, sigma = {self.K_prior[1]}")
                self.logger.info(f"  - v0: mean = {self.v0_prior[0]}, sigma = {self.v0_prior[1]}") 
                self.logger.info(f"  - p: mean = {self.p_prior[0]}, sigma = {self.p_prior[1]}")

        #--------------------------------
        # Start optimization
        #--------------------------------

        # If time is present, we flatten the sequence and time dimensions to have a single sequence dimension
        # but compute the indices of the new flattened sequences to be able to recover the original shape of 
        # the object at the end of the optimization        
        n_seq = self.n_seq * self.n_t
        ind = np.arange(n_seq)
        n_seq_total = n_seq
                        
        ind = self.split_and_merge(n_seq, self.n_t, simultaneous_sequences)
        
        n_sequences = len(ind)
        
        self.modes_seq = [None] * n_sequences
        self.pars_s0_seq = [None] * n_sequences
        self.pars_s2_seq = [None] * n_sequences
        self.pars_raw_seq = [(None, None)] * n_sequences
        self.jitter_seq = [None] * n_sequences
        self.shift_seq = [None] * n_sequences
        self.loss = [None] * n_sequences

        self.psf_seq = [None] * n_sequences
        self.degraded_seq = [None] * n_sequences
        self.obj_seq = [None] * n_sequences
        self.obj_diffraction_seq = [None] * n_sequences
        self.obj_infer_seq = [None] * n_sequences
        self.n_active_seq = [None] * n_sequences
        self.failed_batches = []

        tinit = time.time()
        tinit_global = time.time()

        self.total_time_convergence = 0.0

        if self.psf_model.lower() in ['zernike', 'kl', 'nmf']:
            self.psf_diffraction, self.psf_diffraction_ft = self.compute_psf_diffraction()
        
        for i_seq, seq in enumerate(ind):

            # Restriction of the wavefront of this batch, if a first pass computed one
            self.nullspace_P_batch = self.nullspace_P_seq[i_seq].to(self.device) if self.nullspace_P_seq is not None else None

            # Neighbouring patches of this batch, as positions in the batch, for the coupling regularization
            self.coupling_E_batch = None
            if i_seq == 0 and self.coupling_lambda > 0 and self.coupling_pairs is None:
                self.logger.warning("regularization coupling is set but no neighbours were given (set_neighbours): no coupling")
            if self.coupling_lambda > 0 and self.coupling_pairs is not None and self.psf_model.lower() in ['zernike', 'kl']:
                # With time, sequences are flattened as (s t): sequence s at step t is s * n_t + t. The pairs name
                # patches (s); each pair is coupled within every time step, not across steps (the seeing of different
                # steps is independent)
                pairs = [(p * self.n_t + t, q * self.n_t + t) for p, q in self.coupling_pairs for t in range(self.n_t)]
                pos = {int(g): k for k, g in enumerate(seq)}
                E = [(pos[p], pos[q]) for p, q in pairs if p in pos and q in pos]
                n_cut = sum((p in pos) != (q in pos) for p, q in pairs)
                if n_cut:
                    self.logger.warning(f"coupling: {n_cut} pairs of neighbours fall in different batches and are not coupled; "
                                        f"raise simultaneous_sequences to fit all patches together")
                if E:
                    self.coupling_E_batch = torch.tensor(E, dtype=torch.long, device=self.device)
            
            if i_seq == 0:
                label_time = ''
            else:
                deltat = tfinal - tinit
                tinit = time.time()
                remaining = deltat * (n_sequences - i_seq)
                label_time = f" - Elapsed time {deltat:.2f} s - Remaining time {remaining:.2f} s ({remaining:.2f} s)"
                
            if len(seq) > 1:
                self.logger.info(f"Processing sequences [{seq[0]+1},{seq[-1]+1}]/{n_seq_total} {label_time}")
            else:
                self.logger.info(f"Processing sequence {seq[0]+1}/{n_seq_total} {label_time}")

            frames_apodized_seq = []
            plane_seq = []
            frames_ft = []
            sigma_seq = []
            diversity_seq = []
            for i in range(self.n_o):
                frames_apodized_seq.append(self.frames_apodized[i][seq, ...].to(self.device))
                plane_seq.append(self.plane[i][seq, ...].to(self.device))
                frames_ft.append(self.frames_ft[i][seq, ...].to(self.device))
                sigma_seq.append(self.sigma[i][seq, ...].to(self.device))
                diversity_seq.append(self.diversity[i][seq, ...].to(self.device))

            n_seq = len(seq)

            # Quantities of the time path that depend only on the frames of this batch (compute_object)
            self._time_constants = {}

            # Objects whose PSFs are those of the first object: same pupil, basis and defocus (the same wavelength), the
            # same frames per diversity channel and the same diversity in this batch, no jitter (compute_psfs)
            self._psf_same_as_first = None
            if self.n_o > 1 and self.use_jitter == 'none' and self.psf_model.lower() in ['zernike', 'kl']:
                self._psf_same_as_first = [i > 0 and self.pupil[i] is self.pupil[0] and self.basis[i] is self.basis[0]
                                           and self.defocus_basis[i] is self.defocus_basis[0]
                                           and list(self.init_frame_diversity[i]) == list(self.init_frame_diversity[0])
                                           and torch.equal(diversity_seq[i], diversity_seq[0]) for i in range(self.n_o)]
                                                
            if (infer_object):
                
                obj_init = [None] * self.n_o

                for i in range(self.n_o):

                    if obj_in is not None:
                        # obj_in: one tensor for all objects or a list with one per object, each
                        # (nx, ny), or (n_sequences, nx, ny) with n_sequences = n_seq * n_t
                        self.logger.info(f"Using provided initial object...")
                        tmp = obj_in[i] if isinstance(obj_in, (list, tuple)) else obj_in
                        tmp = torch.as_tensor(tmp, dtype=torch.float32)
                        if tmp.ndim == 2:
                            tmp = tmp[None, :, :].expand(n_seq, -1, -1)
                        elif tmp.shape[0] == n_seq_total and n_seq_total != n_seq:
                            tmp = tmp[seq, ...]
                        if tuple(tmp.shape) != (n_seq, self.n_x, self.n_y):
                            raise ValueError(f"obj_in has shape {tuple(tmp.shape)}; expected ({self.n_x}, {self.n_y}) or ({n_seq_total}, {self.n_x}, {self.n_y})")
                        obj_init[i] = tmp.to(self.device)
                    else:
                        # Frames of the first diversity channel, (n_seq, n_f, nx, ny)
                        frames_init = frames_apodized_seq[i][:, 0:self.n_f, :, :]
                        if self.config['initialization']['object'] == 'contrast':
                            self.logger.info(f"Selecting initial object as image with best contrast...")
                            contrast = torch.std(frames_init, dim=(-1, -2)) / torch.mean(frames_init, dim=(-1, -2))
                            best = torch.argmax(contrast, dim=1)
                            obj_init[i] = frames_init[torch.arange(n_seq), best, :, :]
                        if self.config['initialization']['object'] == 'average':
                            self.logger.info(f"Selecting initial object as average image...")
                            obj_init[i] = torch.mean(frames_init, dim=1)
                    
                    # Initialize the object with the inverse of softplus(x * scale) / scale; the floor
                    # keeps pixels at or near zero finite (log(expm1(z)) -> -inf as z -> 0)
                    if (self.config['optimization']['transform'] == 'softplus'):
                        scale = self.config['optimization']['softplus_scale']
                        z = torch.clamp(obj_init[i] * scale, min=1e-6)
                        obj_init[i] = (z + torch.log(-torch.expm1(-z))) / scale
            
            # Unknown modes
            if modes_in is not None:
                self.logger.info(f"Using provided initial modes...")
                modes = modes_in.clone().detach().to(self.device).requires_grad_(True)
            else:

                # Unknowns for ZERNIKE/KL modes
                if self.psf_model.lower() in ['zernike', 'kl']:
                    if self.config['initialization']['modes_std'] == 0:
                        self.logger.info(f"Initializing modes with zeros...")
                        modes = torch.zeros((n_seq, self.n_f, self.n_modes), device=self.device, requires_grad=True)
                    else:
                        self.logger.info(f"Initializing modes with random values with standard deviation {self.config['initialization']['modes_std']}")
                        modes = self.config['initialization']['modes_std'] * torch.randn((n_seq, self.n_f, self.n_modes))
                        modes = modes.clone().detach().to(self.device).requires_grad_(True)

                # Unknowns for NMF modes
                if self.psf_model.lower() == 'nmf':                               
                    init_mode = self.config.get('initialization', {}).get('modes', 'diffraction')
                    if init_mode == 'cross_spectrum_nnls':
                        self.logger.info("Initializing NMF modes using Cross-Spectrum NNLS...")
                        
                        # Use precomputed uncentered diffraction OTF as baseline reference
                        otf_ref = self.psf_diffraction_ft[0]
                        
                        # Initialize target arrays for the coefficients (shape: n_seq, n_f, n_modes)
                        basis_flat = self.basis[0].reshape((self.n_modes, self.npix**2)) # (n_modes, npix**2)
                        
                        # Collect all target PSFs into a single tensor
                        target_psfs = torch.zeros((n_seq, self.n_f, self.npix, self.npix), device=self.device)
                        
                        # Process per sequence batch and frame to get target PSFs
                        for i_s in range(n_seq):
                            frames_seq = frames_apodized_seq[0][i_s]  # (n_f, npix, npix)
                            frames_ft_seq = torch.fft.fft2(frames_seq, norm=self.fft_norm)  # (n_f, npix, npix)
                            mean_frame_ft = torch.mean(frames_ft_seq, dim=0)  # (npix, npix)
                            
                            # Stabilized division denominator
                            den = torch.conj(mean_frame_ft) * mean_frame_ft
                            mean_den = torch.mean(den).real
                            epsilon = 1e-4 * mean_den + 1e-12
                            
                            for i_f in range(self.n_f):
                                # Ratio of frame OTF to mean frame OTF
                                ratio = (frames_ft_seq[i_f] * torch.conj(mean_frame_ft)) / (den + epsilon)
                                target_otf = ratio * otf_ref
                                target_psf = torch.fft.ifft2(target_otf, norm=self.fft_norm).real
                                target_psfs[i_s, i_f, :, :] = torch.fft.fftshift(target_psf, dim=[-2, -1])
                        
                        # Flatten targets to shape (n_seq * n_f, npix**2)
                        Y = target_psfs.reshape((n_seq * self.n_f, self.npix**2))
                        
                        # Precompute quadratic terms for batched NNLS
                        # H = A * A^T (n_modes, n_modes)
                        H = torch.matmul(basis_flat, basis_flat.t())
                        # G = Y * A^T (n_seq * n_f, n_modes)
                        G = torch.matmul(Y, basis_flat.t())
                        
                        # Compute Lipschitz constant (largest eigenvalue of H) for optimal step size
                        L = torch.linalg.eigvalsh(H)[-1].item()
                        step = 1.0 / L
                        
                        # Batched Accelerated Projected Gradient (APG) for NNLS
                        X = torch.zeros((n_seq * self.n_f, self.n_modes), device=self.device)
                        Y_apg = X.clone()
                        t_apg = 1.0
                        for _ in range(80):
                            # Gradient step & non-negative projection
                            X_next = torch.clamp(Y_apg - step * (torch.matmul(Y_apg, H) - G), min=0.0)
                            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_apg**2))
                            Y_apg = X_next + ((t_apg - 1.0) / t_next) * (X_next - X)
                            X = X_next
                            t_apg = t_next
                            
                        tmp_torch = X.reshape((n_seq, self.n_f, self.n_modes))
                    else:
                        # Centering the diffraction PSF to match the NMF basis spatial layout
                        self.logger.info("Initializing NMF modes using Diffraction PSF NNLS fit...")
                        psf_diff_centered = torch.fft.fftshift(self.psf_diffraction[0], dim=[-2, -1])
                        tmp, _ = optim.nnls(self.basis[0].reshape((self.n_modes, self.npix**2)).T.cpu().numpy(), psf_diff_centered.flatten().cpu().numpy())
                        tmp_torch = torch.tensor(tmp.astype('float32'))[None, None, :].repeat((n_seq, self.n_f, 1))
                    
                    # Convert physical NNLS coefficients to optimization variables
                    # account for softplus_scale: x = inv_softplus(tmp * scale) / scale
                    scale = self.config['optimization'].get('softplus_scale', 1.0)
                    
                    # Safe inverse softplus: for large values x ~= y, for small values use the log formula
                    threshold = 20.0
                    modes_val = torch.where(tmp_torch * scale > threshold, 
                                          tmp_torch, 
                                          torch.log(torch.exp(tmp_torch * scale) - 1.0 + 1e-10) / scale)
                    
                    modes = modes_val.clone().detach().to(self.device).requires_grad_(True)
                    
                    # Initialize learnable PSF shift (one per sequence, in pixels) if enabled in config
                    if self.config['psf']['shift']:
                        self.shift_x = torch.zeros((n_seq, self.n_f), device=self.device, requires_grad=True)
                        self.shift_y = torch.zeros((n_seq, self.n_f), device=self.device, requires_grad=True)

                # Unknowns for VAE modes: latents ~ initialization.modes_std * N(0, 1), zeros for modes_std = 0
                if self.psf_model.lower() == 'vae':
                    std = self.config['initialization']['modes_std']
                    self.logger.info(f"Initializing VAE latents with standard deviation {std}...")
                    modes = std * torch.randn((n_seq, self.n_f, self.n_modes))
                    modes = modes.to(self.device).requires_grad_(True)

            # VAE tip-tilt (also when the latents come from modes_in)
            if self.psf_model.lower() == 'vae':
                shift_x = torch.zeros((n_seq, self.n_f), device=self.device, requires_grad=True)
                shift_y = torch.zeros((n_seq, self.n_f), device=self.device, requires_grad=True)
                    
            if self.psf_model.lower() == 'nmf' and self.config['psf']['shift']:
                self.logger.info(f"NMF: adding learnable PSF shift (shift_x, shift_y) to optimizer...")

            if (infer_object):
                self.logger.info(f"Optimizing object and modes...")

                parameters = [{'params': modes, 'lr': self.lr_modes}]
                if self.psf_model.lower() == 'nmf' and self.config['psf']['shift']:
                    parameters.append({'params': self.shift_x, 'lr': self.lr_modes})
                    parameters.append({'params': self.shift_y, 'lr': self.lr_modes})
                if self.psf_model.lower() == 'vae' and self.config['psf']['shift']:
                    parameters.append({'params': shift_x, 'lr': self.lr_modes})
                    parameters.append({'params': shift_y, 'lr': self.lr_modes})
                obj = [None] * self.n_o

                for i in range(self.n_o):
                    obj[i] = obj_init[i].clone().detach().to(self.device).requires_grad_(True)
                    parameters.append({'params': obj[i], 'lr': self.lr_obj})
                                    
            else:
                self.logger.info(f"Optimizing modes only...")

                parameters = [{'params': modes, 'lr': self.lr_modes}]
                if self.psf_model.lower() == 'nmf' and self.config['psf']['shift']:
                    parameters.append({'params': self.shift_x, 'lr': self.lr_modes})
                    parameters.append({'params': self.shift_y, 'lr': self.lr_modes})

                if self.psf_model.lower() == 'vae' and self.config['psf']['shift']:
                    parameters.append({'params': shift_x, 'lr': self.lr_modes})
                    parameters.append({'params': shift_y, 'lr': self.lr_modes})

                if self.loss_type == 'marginal':
                    if 'psd' in self.config:
                        # Upstream format: pars_s0 shape (n_o, 3), pars_s2 shape (n_o, 1)
                        pars_s0 = np.zeros((self.n_o, 3))
                        for i in range(self.n_o):
                            pars_s0[i, 0] = np.log(self.K_prior[0] / self.npix)
                            pars_s0[i, 1] = np.log(self.v0_prior[0])
                            pars_s0[i, 2] = np.log(self.p_prior[0])
                        self.pars_s0_out = np.zeros((self.n_o, 3))
                        pars_s0_torch = torch.tensor(pars_s0.astype('float32')).to(self.device).requires_grad_(True)
                        self.pars_s0_out = torch.tensor(self.pars_s0_out.astype('float32')).to(self.device)
                        if self.lr_prior > 0:
                            parameters.append({'params': pars_s0_torch, 'lr': self.lr_prior})

                        pars_s2 = np.zeros((self.n_o, 1))
                        for i in range(self.n_o):                        
                            pars_s2[i, 0] = np.log(sigma_seq[i].mean().item()**2)
                        self.pars_s2_out = np.zeros((self.n_o, 1))
                        pars_s2_torch = torch.tensor(pars_s2.astype('float32')).to(self.device).requires_grad_(True)
                        self.pars_s2_out = torch.tensor(self.pars_s2_out.astype('float32')).to(self.device)
                        if self.lr_prior > 0:
                            parameters.append({'params': pars_s2_torch, 'lr': self.lr_prior})
                    else:
                        # Local format fallback: pars_s0 shape (n_seq, n_o, 4)
                        pars_s0 = np.zeros((n_seq, self.n_o, 4))
                        for i in range(self.n_o):
                            pars_s0[:, i, 0] = np.log(self.K_prior[0])
                            pars_s0[:, i, 1] = np.log(self.v0_prior[0])
                            pars_s0[:, i, 2] = np.log(self.p_prior[0])     # get_su_s2 reads p = exp(.)
                            pars_s0[:, i, 3] = np.log(sigma_seq[i].mean().item()**2)
                        self.pars_s0_out = np.zeros((n_seq, self.n_o, 4))
                        pars_s0_torch = torch.tensor(pars_s0.astype('float32')).to(self.device).requires_grad_(True)
                        self.pars_s0_out = torch.tensor(self.pars_s0_out.astype('float32')).to(self.device)
                        pars_s2_torch = None
                        if self.lr_prior > 0:
                            parameters.append({'params': pars_s0_torch, 'lr': self.lr_prior})
                        # the noise variance of this format is pars_s0[..., 3]; no separate tensor

            # Fast jitter, faster than the integration time (both branches: the closure reads jitter_torch)
            if self.use_jitter == 'fast':
                jitter = np.zeros((n_seq, self.n_f, 3))
                jitter[:, :, 0] = -2.0
                jitter[:, :, 1] = -2.0
                jitter[:, :, 2] = 0.01

                jitter_torch = torch.tensor(jitter.astype('float32')).to(self.device).requires_grad_(True)
                if self.lr_modes > 0:
                    parameters.append({'params': jitter_torch, 'lr': self.lr_modes})

            # Slow jitter, parameterized in terms of a Bezier curve with 3 ccontrol points
            # The first one is always at (0, 0)
            if self.use_jitter == 'slow':
                jitter = 0.3*np.random.randn(n_seq, self.n_f, 2, 2)
                jitter = np.zeros((n_seq, self.n_f, 2, 2))
                
                jitter_torch = torch.tensor(jitter.astype('float32')).to(self.device).requires_grad_(True)
                parameters.append({'params': jitter_torch, 'lr': self.lr_modes})

            # Second order optimizer
            if optimizer == 'lbfgs':
                self.logger.info(f"Using LBFGS optimizer...")
                # LBFGS does not support parameter groups (different LRs for different params)
                # We flatten the parameters into a single list
                params_lbfgs = []
                for group in parameters:
                    if isinstance(group, dict):
                        params_lbfgs.append(group['params'])
                    else:
                        params_lbfgs.append(group)
                opt = torch.optim.LBFGS(params_lbfgs, lr=self.lr_modes, line_search_fn='strong_wolfe', max_iter=self.lbfgs_max_iter)
            if optimizer == 'adam':
                self.logger.info(f"Using Adam optimizer...")
                opt = torch.optim.Adam(parameters)
            if optimizer == 'adamw':
                self.logger.info(f"Using AdamW optimizer...")
                opt = torch.optim.AdamW(parameters)
            if optimizer == 'cg':
                if NGC_OPTIMIZER:
                    self.logger.info(f"Using CG optimizer...")
                    opt = ncg_optimizer.BASIC(parameters, method = 'CD', line_search = 'Strong_Wolfe', c1 = 1e-4, c2 = 0.9, lr = 1, rho = 0.5, eps=1e-8)
                else:
                    self.logger.info(f"Using Adam optimizer...")
                    opt = torch.optim.Adam(parameters)

            if optimizer not in ['lbfgs', 'adam', 'adamw', 'cg']:
                raise ValueError(f"Optimizer {optimizer} not supported")
            
            self.logger.info(f"Starting optimization with {n_iterations} iterations...")
            self.logger.info(f"Learning rates: modes {self.lr_modes} - object {self.lr_obj if infer_object else 'N/A'} - psd prior {self.lr_prior if not infer_object and self.loss_type == 'marginal' else 'N/A'}")
                
            losses = torch.zeros(n_iterations, device=self.device)

            t = tqdm(range(n_iterations))

            # The vae fits all latents in the first step and follows the schedule afterwards: started at the
            # schedule's first value with latents initialised at zero it settles on too broad a PSF.
            if self.psf_model.lower() in ['zernike', 'kl', 'nmf']:
                n_active = int(self.anneal[0])
            else:
                n_active = self.n_modes

            modes_previous = modes.clone().detach()

            self.t0_convergence = time.time()

            flip = 0

            # Last parameters at which the loss was finite, restored if the optimisation diverges
            last_good = [p.detach().clone() for g in opt.param_groups for p in g['params']]
            psd_frozen = False
            n_small, n_active_previous = 0, -1
                
            for loop in t:
                                
                def closure(backward=True):
                            
                    if backward:
                        opt.zero_grad(set_to_none=True)

                    if self.psf_model.lower() in ['zernike', 'kl']:
                
                        # Compute PSF from current wavefront coefficients, tip-tilt referenced to its mean over the
                        # frames (as MOMFBD): a common shift of all PSFs is a shift of the object, and anchoring it to
                        # one frame would put that frame's local image motion into the object of each patch
                        modes_centered = modes.clone()
                        modes_centered[:, : , 0:2] = modes_centered[:, :, 0:2] - modes[:, :, 0:2].mean(dim=1, keepdim=True)

                        if self.remove_tt:
                            modes_centered[:, :, 0:2] = 0.0
                                    
                        # modes -> (n_seq, n_f, self.n_modes)
                        # jitter > (n_seq, n_f, 3)                                                
                        # Half spectrum for the joint wiener loss (compute_loss_half); the object for the progress bar
                        # or an object regularization needs the full OTFs
                        use_half = (self.half_spectrum and self.loss_filter == 'wiener' and self.loss_type == 'joint'
                                    and self.use_jitter == 'none' and not infer_object
                                    and not (self.show_object_info or len(self.index_regularization['object']) > 0))
                        # half is passed only when used, so that subclasses and wrappers of compute_psfs keep working
                        psf, otf = self.compute_psfs(modes_centered[..., 0:n_active], diversity_seq,
                                                     jitter=jitter_torch if self.use_jitter != 'none' else None,
                                                     **({'half': True} if use_half else {}))

                    # Coupling of neighbouring patches: lambda sum_pairs,frames ||theta_p - theta_q||^2 over the first
                    # coupling_modes active coefficients, normalised as the joint loss (sequences x (npix - 1)^2)
                    loss_coupling = None
                    if self.coupling_E_batch is not None:
                        m = modes_centered[..., 0:min(self.coupling_modes, n_active)]
                        E = self.coupling_E_batch
                        loss_coupling = self.coupling_lambda * torch.sum((m[E[:, 0]] - m[E[:, 1]])**2) / (m.shape[0] * (self.npix - 1)**2)
                    
                    if self.psf_model.lower() == 'nmf':
                        psf, otf = self.compute_psfs_nmf(modes[..., 0:n_active])

                    if self.psf_model.lower() == 'vae':

                        # Remove the tip-tilt of the first frame of each sequence to avoid degeneracy in the optimization
                        shift_x_centered = shift_x.clone()
                        shift_y_centered = shift_y.clone()
                        shift_x_centered = shift_x_centered - shift_x[:, 0:1]
                        shift_y_centered = shift_y_centered - shift_y[:, 0:1]
                        psf, otf = self.compute_psfs_vae(modes[..., 0:n_active], shift_x=shift_x_centered, shift_y=shift_y_centered)
                    
                    if (infer_object):

                        # Object (softplus(obj * softplus_scale) / softplus_scale or clamped) and its spectrum
                        obj_ft, obj_filter_ft, obj_filter = self.object_from_parameters(obj)

                        loss_data = torch.tensor(0.0).to(self.device)

                        for i in range(self.n_o):

                            degraded_ft = obj_ft[i][:, None, :, :] * otf[i]

                            # Chi^2 with per-frame noise variances: 0.5 * sum_j |D_j - O S_j|^2 / sigma_j^2,
                            # averaged over sequences and frequencies (the joint data term without the
                            # object is eliminated). With the ortho FFT the noise variance per frequency
                            # is sigma^2, so the loss is O(n_f) at the solution and Adam's eps is irrelevant.
                            residual = degraded_ft - frames_ft[i]
                            weight = 1.0 / (sigma_seq[i][:, :, None, None]**2 + 1e-20)
                            loss_data = loss_data + 0.5 * torch.mean(torch.sum(weight * (residual * torch.conj(residual)).real, dim=1))

                        loss_prior = torch.tensor(0.0).to(self.device)
                        loss = loss_data + loss_prior

                    else:                        

                        pars_s0_arg = pars_s0_torch if self.loss_type == 'marginal' else None
                        pars_s2_arg = pars_s2_torch if (self.loss_type == 'marginal' and 'psd' in self.config) else None

                        # The object is needed for the progress bar and for any object regularization
                        if self.show_object_info or len(self.index_regularization['object']) > 0:
                            obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, 
                                                                                    otf, 
                                                                                    sigma_seq, 
                                                                                    plane_seq, 
                                                                                    pars_s0=pars_s0_arg, 
                                                                                    pars_s2=pars_s2_arg)

                        # Compute the loss function using the appropriate filter for the loss
                        if self.psf_model.lower() in ['zernike', 'kl'] and use_half:
                            loss_data, loss_prior, loss = self.compute_loss_half(frames_ft, otf, sigma_seq)
                        else:
                            loss_data, loss_prior, loss = self.compute_loss(frames_ft,
                                                                            otf,
                                                                            sigma_seq,
                                                                            pars_s0=pars_s0_arg,
                                                                            pars_s2=pars_s2_arg,
                                                                            jitter=jitter_torch if self.use_jitter != 'none' else None)

                    if loss_coupling is not None:
                        loss = loss + loss_coupling

                    # If MOMFBD is used, then the object cannot be regularized. Look for alternatives for the future
                    # Object regularization
                    loss_obj = torch.tensor(0.0).to(self.device)
                    for index in self.index_regularization['object']:
                        loss_obj += self.regularization[index](obj_filter)
                    
                    # Total loss
                    loss += loss_obj
                                                        
                    # Save some information for the progress bar
                    # self.loss_local = loss.detach()
                    # self.obj_filter = [None] * self.n_o
                    # for i in range(self.n_o):
                    #     self.obj_filter[i] = obj_filter[i].detach()
                    # self.loss_mse_local = loss_mse.detach()
                    # self.loss_obj_local = loss_obj.detach()

                    if backward:
                        if optimizer == 'cg' and NGC_OPTIMIZER:
                            loss.backward(retain_graph=True)
                        else:
                            loss.backward()

                    self.loss_data_local = loss_data.detach()
                    self.loss_prior_local = loss_prior.detach()
                    self.loss_local = loss.detach()
                    self.loss_obj_local = loss_obj.detach()

                    if self.show_object_info:
                        self.obj_filter_local = obj_filter
                    
                    return loss
                        
                # Number of modes the closure fits in this step; the exported PSF uses the last one
                n_active_fit = n_active

                params_step = [p for g in opt.param_groups for p in g['params']]
                snapshot = [p.detach().clone() for p in params_step]

                loss = opt.step(closure)

                # LBFGS returns the loss of the step's first evaluation and leaves the last line-search trial
                # in the progress fields: evaluate the accepted point instead (forward only)
                if optimizer == 'lbfgs':
                    with torch.no_grad():
                        loss = closure(backward=False)

                # Adam's loss belongs to the parameters before the step, LBFGS's (above) to those after it
                if torch.isfinite(loss):
                    last_good = [p.detach().clone() for p in params_step] if optimizer == 'lbfgs' else snapshot

                if not torch.isfinite(loss) or not all(bool(torch.isfinite(p).all()) for p in params_step):
                    with torch.no_grad():
                        for p, p_good in zip(params_step, last_good):
                            p.copy_(p_good)
                    msg = (f"Non-finite loss or parameters at iteration {loop} of sequence batch {i_seq}: "
                           f"parameters restored to the last finite point and the optimisation of this batch stopped")
                    self.logger.warning(msg)
                    warnings.warn(msg, RuntimeWarning)
                    self.failed_batches.append(i_seq)
                    losses[loop:] = float('nan')
                    break

                losses[loop] = loss.detach()

                # Stopping rule, armed once the schedule has reached its final number of modes
                if self.tolerance is not None:
                    if n_active_fit == self.anneal[-1] and loop > 0 and torch.isfinite(losses[loop - 1]) and \
                            n_active_previous == n_active_fit:
                        rel_change = float(torch.abs(losses[loop] - losses[loop - 1]) / (torch.abs(losses[loop - 1]) + 1e-30))
                        n_small = n_small + 1 if rel_change < self.tolerance else 0
                    else:
                        n_small = 0
                    n_active_previous = n_active_fit
                    if n_small >= self.patience:
                        self.logger.info(f"Converged at iteration {loop}: relative change below tolerance {self.tolerance} for {self.patience} iterations")
                        losses = losses[0:loop + 1]
                        break

                loss_data = self.loss_data_local
                loss_prior = self.loss_prior_local
                loss = self.loss_local
                loss_obj = self.loss_obj_local
                
                if self.show_object_info:
                    obj_filter = self.obj_filter_local

                # Reduce the learning rate of the PSD parameters
                # if self.loss_type == 'marginal':
                    # opt.param_groups[1]['lr'] *= 0.99                                
                
                tmp = OrderedDict()

                # handle is None on CPU and on CUDA without nvitop
                if self.cuda and self.handle is not None:
                    gpu_usage = f'{self.handle.gpu_utilization():03d}'
                    memory_usage = f'{self.handle.memory_used() / 1024**2:4.1f}/{self.handle.memory_total() / 1024**2:4.1f} MB'
                    tmp['gpu'] = f'{gpu_usage} %'
                    tmp['mem'] = f'{memory_usage}'

                delta_modes = torch.mean(torch.abs(modes.detach() - modes_previous))

                modes_previous = modes.clone().detach()

                tmp['active'] = f'{n_active}'
                if self.show_object_info:
                    tmp['contrast'] = f'{torch.std(obj_filter[0]) / torch.mean(obj_filter[0]) * 100.0:7.4f}'
                    tmp['minmax'] = f'{torch.min(obj_filter[0]):7.4f}/{torch.max(obj_filter[0]):7.4f}'
                tmp['chg'] = f'{delta_modes.item():8.6f}'
                if self.loss_type == 'marginal' and len(opt.param_groups) > 1:
                    tmp['lr'] = f'{opt.param_groups[0]["lr"]:6.3f}/{opt.param_groups[1]["lr"]:6.3f}'
                else:
                    tmp['lr'] = f'{opt.param_groups[0]["lr"]:6.3f}'
                tmp['LDATA'] = f'{loss_data.detach().item():8.6f}'
                tmp['LPRIOR'] = f'{loss_prior.detach().item():8.6f}'
                if self.loss_type == 'marginal' and self.pars_s0_out.ndim == 2:
                    tmp['K'] = f'{self.pars_s0_out[0][0].item() * self.npix:6.2f}'
                    tmp['v0'] = f'{self.pars_s0_out[0][1].item():6.2f}'
                    tmp['p'] = f'{self.pars_s0_out[0][2].item():6.2f}'
                    tmp['sig'] = f'{np.sqrt(self.pars_s2_out[0][0].item()):7.4f}'
                if self.loss_type == 'marginal' and self.pars_s0_out.ndim == 3:
                    # per-sequence format (n_seq, n_o, 4): K, v0, p, s2 of object 0, averaged over the batch
                    tmp['K'] = f'{self.pars_s0_out[:, 0, 0].mean().item():6.2f}'
                    tmp['v0'] = f'{self.pars_s0_out[:, 0, 1].mean().item():6.2f}'
                    tmp['p'] = f'{self.pars_s0_out[:, 0, 2].mean().item():6.2f}'
                    tmp['sig'] = f'{np.sqrt(self.pars_s0_out[:, 0, 3].mean().item()):7.4f}'
                if self.use_jitter == 'fast':
                    tmp['sx'] = f'{torch.min(torch.exp(jitter_torch[:, :, 0])).item():6.3f}'
                    tmp['sy'] = f'{torch.min(torch.exp(jitter_torch[:, :, 1])).item():6.3f}'
                    tmp['rxy'] = f'{torch.min(torch.tanh(jitter_torch[:, :, 2])).item():6.3f}'
                if self.use_jitter == 'slow':
                    tmp['px'] = f'{torch.min((jitter_torch[:, :, :, 0])).item():6.3f}/{torch.max((jitter_torch[:, :, :, 0])).item():6.3f}'
                    tmp['py'] = f'{torch.min((jitter_torch[:, :, :, 1])).item():6.3f}/{torch.max((jitter_torch[:, :, :, 1])).item():6.3f}'
                tmp['L'] = f'{loss.detach().item():8.6f}'
                t.set_postfix(ordered_dict=tmp)
                
                n_active = self.anneal[loop]

                # Freeze the PSD hyper-parameters (K, v0, p and the noise variance) once enough modes are active
                if self.loss_type == 'marginal' and not infer_object and not psd_frozen and n_active > self.stop_psd:
                    psd_frozen = True
                    frozen = [pars_s0_torch]
                    if optimizer == 'lbfgs':
                        # LBFGS keeps all parameters in one group: rebuild it without the frozen ones
                        opt = torch.optim.LBFGS([p for p in params_lbfgs if all(p is not f for f in frozen)],
                                                lr=self.lr_modes, line_search_fn='strong_wolfe', max_iter=self.lbfgs_max_iter)
                    else:
                        for g in opt.param_groups:
                            if any(p is f for p in g['params'] for f in frozen):
                                g['lr'] = 0.0
                    for f in frozen:
                        f.requires_grad_(False)

                
            self.tf_convergence = time.time()

            self.total_time_convergence += self.tf_convergence - self.t0_convergence
            
            # PSFs exactly as in the last closure: same number of active modes, tip-tilt removed if
            # psf.remove_tt, VAE shifts referenced to frame 0
            with torch.no_grad():
                psf, otf = self.psfs_from_parameters(modes.detach(), diversity_seq, n_active_fit,
                                                     jitter=jitter_torch.detach() if self.use_jitter != 'none' else None,
                                                     shift=(shift_x.detach(), shift_y.detach()) if self.psf_model.lower() == 'vae' else None)
                
            if (infer_object):
                
                # Filtered object from the final estimate, with the apodization gradient added back
                with torch.no_grad():
                    obj_ft, obj_filter_ft, obj_filter = self.object_from_parameters(obj, plane_seq)
                self.obj_infer_seq[i_seq] = [o.detach() for o in obj]

            else:
                pars_s0_arg = pars_s0_torch if self.loss_type == 'marginal' else None
                pars_s2_arg = pars_s2_torch if (self.loss_type == 'marginal' and 'psd' in self.config) else None
                obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, otf, sigma_seq, plane_seq, pars_s0=pars_s0_arg, pars_s2=pars_s2_arg)
                                   

            obj_filter_diffraction = [None] * self.n_o
            degraded = [None] * self.n_o
            for i in range(self.n_o):                
                if self.psf_model.lower() in ['zernike', 'kl']:
                    obj_filter_diffraction[i] = torch.fft.ifft2(obj_filter_ft[i] * self.psf_diffraction_ft[i][None, :, :], norm=self.fft_norm).real
            
                # Compute final degraded images
                degraded_ft = obj_filter_ft[i][:, None, :, :] * otf[i]
                # obj_filter_ft is the unnormalised DFT of the object and otf the ortho OTF, so the product
                # is the ortho spectrum of the degraded image, so it is inverted with the same norm
                degraded[i] = torch.fft.ifft2(degraded_ft, norm=self.fft_norm).real
            
            # Store the results for the current set of sequences
            self.modes_seq[i_seq] = modes.detach()
            self.n_active_seq[i_seq] = n_active_fit
            self.jitter_seq[i_seq] = jitter_torch.detach() if self.use_jitter != 'none' else None
            if self.psf_model.lower() == 'nmf' and self.config['psf']['shift']:
                self.shift_seq[i_seq] = (self.shift_x.detach(), self.shift_y.detach())
            if self.psf_model.lower() == 'vae':
                self.shift_seq[i_seq] = (shift_x.detach(), shift_y.detach())
            self.pars_s0_seq[i_seq] = self.pars_s0_out if not infer_object and self.loss_type == 'marginal' else None
            self.pars_s2_seq[i_seq] = self.pars_s2_out if not infer_object and self.loss_type == 'marginal' and 'psd' in self.config else None
            # pars_s0_seq and pars_s2_seq hold K, v0, p and s2 themselves; compute_object takes the untransformed
            # parameters, so update_object reuses the ones the final object was computed with
            if not infer_object and self.loss_type == 'marginal':
                self.pars_raw_seq[i_seq] = (pars_s0_torch.detach().clone(),
                                            pars_s2_torch.detach().clone() if 'psd' in self.config else None)
            self.loss[i_seq] = losses.detach()

            for i in range(self.n_o):
                psf[i] = psf[i].detach().cpu()
                degraded[i] = degraded[i].detach().cpu()
                obj_filter[i] = obj_filter[i].detach()
                if self.psf_model.lower() in ['zernike', 'kl']:
                    obj_filter_diffraction[i] = obj_filter_diffraction[i].detach()

            self.psf_seq[i_seq] = psf
            self.degraded_seq[i_seq] = degraded
            self.obj_seq[i_seq] = obj_filter
            self.obj_diffraction_seq[i_seq] = obj_filter_diffraction

            tfinal = time.time()

            # del psf, degraded, obj_filter, obj_filter_diffraction, degraded_ft, obj_ft, obj_filter_ft, otf
        
        self.nullspace_P_batch = None
        self.coupling_E_batch = None
        self._time_constants = None
        self._psf_same_as_first = None

        deltat = tfinal - tinit
        deltat_global = tfinal - tinit_global        
        self.total_time = deltat_global        
        self.logger.info(f"Elapsed time {deltat:.2f} s - Total time: {deltat_global:.2f} s")

        # Concatenate the results from all sequences and all objects independently
        # self.psf = [None] * self.n_o
        # self.degraded = [None] * self.n_o
        self.obj = [None] * self.n_o
        self.obj_diffraction = [None] * self.n_o
        self.pars_s0 = [None] * self.n_o        

        # for i in range(self.n_o):
        self.modes = torch.cat(self.modes_seq, dim=0)
        self.loss = torch.cat(self.loss, dim=0)
        self.jitter = torch.cat(self.jitter_seq, dim=0) if self.use_jitter != 'none' else None
                
        for i in range(self.n_o):
            # tmp = [self.psf_seq[j][i] for j in range(n_sequences)]
            # self.psf[i] = torch.cat(tmp, dim=0)

            # tmp = [self.degraded_seq[j][i] for j in range(n_sequences)]
            # self.degraded[i] = torch.cat(tmp, dim=0)
            
            tmp = [self.obj_seq[j][i] for j in range(n_sequences)]            
            self.obj[i] = torch.cat(tmp, dim=0)

            # Put back time dimension if it was present in the input data
            if self.time_present:
                self.obj[i] = rearrange(self.obj[i], '(s t) h w -> s t h w', t=self.n_t, s=self.n_seq)

            tmp = [self.obj_diffraction_seq[j][i] for j in range(n_sequences)]

            if self.psf_model.lower() in ['zernike', 'kl']:
                self.obj_diffraction[i] = torch.cat(tmp, dim=0)
                if self.time_present:
                    self.obj_diffraction[i] = rearrange(self.obj_diffraction[i], '(s t) h w -> s t h w', t=self.n_t, s=self.n_seq)
            
            if self.loss_type == 'marginal':
                if 'psd' in self.config:
                    tmp_s0 = [self.pars_s0_seq[j][i, :] for j in range(n_sequences)]
                    tmp_s2 = [self.pars_s2_seq[j][i, :] for j in range(n_sequences)]
                    if len(tmp_s0) > 0:
                        self.pars_s0[i] = torch.stack(tmp_s0, dim=0)
                        # We can also store pars_s2 as an attribute if needed
                        self.pars_s2 = [None] * self.n_o
                        self.pars_s2[i] = torch.stack(tmp_s2, dim=0)
                    else:
                        self.pars_s0[i] = None
                else:
                    tmp = [self.pars_s0_seq[j][:, i, :] for j in range(n_sequences)]
                    if len(tmp) > 0:
                        self.pars_s0[i] = torch.cat(tmp, dim=0)
                    else:
                        self.pars_s0[i] = None
                    
        return 
    
    
if __name__ == '__main__': 
    pass

