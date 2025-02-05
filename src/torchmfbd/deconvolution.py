import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as pl
import torchmfbd.zern as zern
import torchmfbd.util as util
from collections import OrderedDict
from tqdm import tqdm
from skimage.morphology import flood
import scipy.ndimage as nd
from nvitop import Device
import logging
import torchmfbd.kl_modes as kl_modes
import torchmfbd.noise as noise
from torchmfbd.reg_smooth import RegularizationSmooth
from torchmfbd.reg_iuwt import RegularizationIUWT
import glob
import pathlib
import yaml
import torchmfbd.configuration as configuration

class Deconvolution(object):
    def __init__(self, config):
        """

        Parameters
        ----------
        npix_apodization : int
            Total number of pixel for apodization (divisible by 2)
        device : str
            Device where to carry out the computations
        batch_size : int
            Batch size
        """
        super().__init__()

        self.logger = logging.getLogger("deconvolution ")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

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

        # Ger handlers to later check memory and usage of GPUs
        if self.cuda:
            if self.config['optimization']['gpu'] < 0:
                self.logger.info(f"GPU is available but not used. Computing in cpu")
                self.cuda = False
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{self.config['optimization']['gpu']}")
                self.handle = Device.all()[self.config['optimization']['gpu']]
                self.logger.info(f"Computing in {self.handle.name()} (free {self.handle.memory_free() / 1024**3:4.2f} GB) - cuda:{self.config['optimization']['gpu']}")
                self.initial_memory_used = self.handle.memory_used()
        else:
            self.logger.info(f"No GPU is available. Computing in cpu")
            self.device = torch.device("cpu")

        # self.n_modes = np.sum(np.arange(self.config['psf']['nmax_modes'])+2)
        self.n_modes = self.config['psf']['nmax_modes']
        self.npix = self.config['images']['n_pixel']
        self.npix_apod = self.config['images']['apodization_border']

        # Get Noll's n order from the number of modes
        # The summation of the modes needs to fulfill n^2+n-2*(k+1)=0
        a = 1.0
        b = 1.0
        c = -2.0 * (self.config['psf']['nmax_modes'] + 1)
        sol1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        sol2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
        n = 0
        if sol1 > 0.0 and sol1.is_integer():
            n = int(sol1)
        if sol2 > 0.0 and sol2.is_integer():
            n = int(sol2)

        self.noll_max = n

        self.psf_model = self.config['psf']['model']

                
        # Generate Hamming window function for WFS correlation
        if (self.npix_apod > 0):
            self.logger.info(f"Using apodization mask with a border of {self.npix_apod} pixels")
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

        # Do some output
        self.logger.info(f"Telescope")        
        self.logger.info(f"  * D: {self.config['telescope']['diameter']} m")
        self.logger.info(f"  * pix: {self.config['images']['pix_size']} arcsec")
        
        # Bookkeeping for objects and diversity
        self.ind_object = []
        self.ind_diversity = []
        self.frames = []
        self.sigma = []
        self.diversity = []

        self.external_regularizations = []
       
    def define_basis(self):

        self.pupil = [None] * self.n_o
        self.basis = [None] * self.n_o
        self.rho = [None] * self.n_o
        self.diffraction_limit = [None] * self.n_o
        self.cutoff = [None] * self.n_o
        self.image_filter = [None] * self.n_o

        # First locate unique wavelengths. We will use the same basis for the same wavelength
        ind_wavelengths = []
        unique_wavelengths = []

        for i in range(self.n_o):
            self.cutoff[i] = self.config[f'object{i+1}']['cutoff']
            self.image_filter[i] = self.config[f'object{i+1}']['image_filter']
            w = self.config[f'object{i+1}']['wavelength']
            if w not in unique_wavelengths:
                unique_wavelengths.append(w)
            
            ind_wavelengths.append(unique_wavelengths.index(w))

        # Normalize wavelengths to scale basis
        unique_wavelengths = np.array(unique_wavelengths)
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
                raise Exception(f"The pixel size is not small enough to model a telescope with D={self.telescope_diameter} cm")

            # Compute telescope aperture
            pupil = util.aperture(npix=self.npix, 
                            cent_obs = self.config['telescope']['central_obscuration'] / self.config['telescope']['diameter'], 
                            spider=0, 
                            overfill=overfill)                

            # PSF model parameterized with the wavefront
            if (self.psf_model.lower() in ['zernike', 'kl']):
                
                self.logger.info(f"PSF model: wavefront expansion")

                if (self.psf_model.lower() not in ['zernike', 'kl']):
                    raise Exception(f"Unknown basis {self.basis}. Use 'zernike' or 'kl' for wavefront expansion")
            
                if (self.psf_model.lower() == 'zernike'):
                    
                    found, filename = self.find_basis_wavefront('zernike', self.config['psf']['nmax_modes'], int(wavelength))

                    # Define Zernike modes        
                    if found:
                        self.logger.info(f"Loading precomputed Zernike {filename}")
                        tmp = np.load(f"{filename}")
                        basis = tmp['basis']           
                    else:                
                        self.logger.info(f"Computing Zernike modes {filename}")
                        basis = self.precalculate_zernike(overfill=overfill)                
                        np.savez(f"{filename}", basis=basis)

                if (self.psf_model.lower() == 'kl'):

                    found, filename = self.find_basis_wavefront('kl', self.config['psf']['nmax_modes'], int(wavelength))

                    if found:
                        self.logger.info(f"Loading precomputed KL basis: {filename}")
                        tmp = np.load(f"{filename}")
                        basis = tmp['basis']                
                    else:
                        self.logger.info(f"Computing KL modes {filename}")
                        self.kl = kl_modes.KL()              
                        
                        basis = self.kl.precalculate(npix_image = self.npix, 
                                        n_modes_max = self.n_modes,                                 
                                        overfill=overfill)
                        np.savez(f"{filename}", basis=basis, variance=self.kl.varKL)
                
                pupil = torch.tensor(pupil.astype('float32')).to(self.device)
                basis = torch.tensor(basis[0:self.n_modes, :, :].astype('float32')).to(self.device)

                # Following van Noort et al. (2005) we normalize the basis
                basis /= normalized_wavelengths[i]

                self.logger.info(f"Wavefront")                        
                self.logger.info(f"  * Using {self.n_modes} modes...")
                    
            # Compute the diffraction limit and the frequency grid
            cutoff = self.config['telescope']['diameter'] / (wavelength * 1e-8) / 206265.0
            freq = np.fft.fftfreq(self.npix, d=self.config['images']['pix_size']) / cutoff
            
            xx, yy = np.meshgrid(freq, freq)
            rho = np.sqrt(xx ** 2 + yy ** 2)

            diffraction_limit = wavelength * 1e-8 / self.config['telescope']['diameter'] * 206265.0

            self.logger.info(f"Wavelength {i} ({wavelength} A)")
            self.logger.info(f"  * Diffraction: {diffraction_limit} arcsec")
            self.logger.info(f"  * Diffraction (x1.22): {1.22 * diffraction_limit} arcsec")

            for j in range(self.n_o):
                if ind_wavelengths[j] == i:
                    self.pupil[j] = pupil
                    self.basis[j] = basis
                    self.rho[j] = rho
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

        for k, v in self.config['regularization'].items():            
            if v['lambda'] != 0.0:
                
                if 'smooth' in k:                
                    tmp = RegularizationSmooth(lambda_reg=v['lambda'], variable=v['variable'])
                if 'iuwt' in k:
                    tmp = RegularizationIUWT(lambda_reg=v['lambda'], variable=v['variable'], nbands=v['nbands'], n_pixel=self.npix)

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
            anneal = (self.noll_max - 2) * (util.sigmoid(a) - util.sigmoid(a + x * (b-a)) ) / ( util.sigmoid(a) - util.sigmoid(b) )
            
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
            self.mask_diffraction[i] = self.rho[i] <= self.cutoff[i]
            self.mask_diffraction_th[i] = torch.tensor(self.mask_diffraction[i].astype('float32')).to(self.device)
            
            # Shifted mask used for the Lofdahl & Scharmer filter
            self.mask_diffraction_shift[i] = np.fft.fftshift(self.mask_diffraction[i])                
                     
    def compute_psfs(self, modes):
        """
        Compute the Point Spread Functions (PSFs) from the given modes.
        Parameters:
        modes (torch.Tensor): A tensor of shape (batch_size, num_modes, height, width) representing the modes.
        Returns:
        tuple: A tuple containing:
            - wavefront (torch.Tensor): The computed wavefronts from the estimated modes.
            - psf_norm (torch.Tensor): The normalized PSFs.
            - psf_ft (torch.Tensor): The FFT of the normalized PSFs.
        """
        

        n_active = modes.shape[2]
                                        
        psf_norm = [None] * self.n_o
        psf_ft = [None] * self.n_o
        
        for i in range(self.n_o):

            # Compute wavefronts from estimated modes                
            wavefront = torch.einsum('ijk,klm->ijlm', modes, self.basis[i][0:n_active, :, :])
            
            # Reuse the same wavefront per object but add the diversity
            wavef = []
            for j in range(len(self.init_frame_diversity[i])):
                div = self.diversity[i][j] * self.basis[i][2, :, :][None, None, :, :]
                wavef.append(wavefront + div)
            
            wavef = torch.cat(wavef, dim=1)
            
            # Compute the complex phase
            phase = self.pupil[i][None, None, :, :] * torch.exp(1j * wavef)

            # Compute FFT of the pupil function and compute autocorrelation
            ft = torch.fft.fft2(phase)
            psf = (torch.conj(ft) * ft).real
            
            # Normalize PSF        
            psf_norm[i] = psf / torch.sum(psf, [-1, -2], keepdim=True)

            # FFT of the PSF
            psf_ft[i] = torch.fft.fft2(psf_norm[i])
        
        return psf_norm, psf_ft
    
    def compute_psf_diffraction(self):
        """
        Compute the Point Spread Functions (PSFs) from diffraction
        
        Returns:
        tuple: A tuple containing:
            - psf_norm (torch.Tensor): The normalized PSFs.
            - psf_ft (torch.Tensor): The FFT of the normalized PSFs.
        """
        
        psf_ft = [None] * self.n_o
        psf_norm = [None] * self.n_o

        for i in range(self.n_o):
            # Compute FFT of the pupil function and compute autocorrelation
            ft = torch.fft.fft2(self.pupil[i])
            psf = (torch.conj(ft) * ft).real
            
            # Normalize PSF        
            psf_norm[i] = psf / torch.sum(psf)

            # FFT of the PSF
            psf_ft[i] = torch.fft.fft2(psf_norm[i])

        return psf_norm, psf_ft
    
    def lofdahl_scharmer_filter(self, Sconj_S, Sconj_I):
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
        den = torch.conj(Sconj_I) * Sconj_I
        H = (Sconj_S / den).real        

        H = torch.fft.fftshift(H).detach().cpu().numpy()

        # noise = 1.35 / np.median(H[:, :, 0:10, 0:10], axis=(2,3))

        H = nd.median_filter(H, [1,1,3,3], mode='wrap')    
        
        filt = 1.0 - H * self.sigma[:, :, None, None].cpu().numpy()**2 * self.npix**2
        filt[filt < 0.2] = 0.0
        filt[filt > 1.0] = 1.0
        
        nb, no, nx, ny = filt.shape

        mask = np.zeros_like(filt)

        for ib in range(nb):
            for io in range(no):
                
                mask[ib, io, :, :] = flood(1.0 - filt[ib, io, :, :], (nx//2, ny//2), tolerance=0.9) * self.mask_diffraction_shift[io, :, :]
                mask[ib, io, :, :] = np.fft.fftshift(mask[ib, io, :, :])

        return torch.tensor(mask).to(Sconj_S.device)

    def compute_object(self, images_ft, psf_ft, sigma, type_filter='tophat'):
        """
        Compute the object in Fourier space using the specified filter.
        Parameters:
        --------
        images_ft (torch.Tensor): 
            The Fourier transform of the observed images.
        psf_ft (torch.Tensor): 
            The Fourier transform of the point spread function (PSF).
        type_filter (str, optional): 
            The type of filter to use ('tophat'/'scharmer'). Default is 'tophat'.
        Returns:
        --------
        torch.Tensor: The computed object in Fourier space.
        """

        out_ft = [None] * self.n_o
        out_filter_ft = [None] * self.n_o
        out_filter = [None] * self.n_o
        
        for i in range(self.n_o):            
                    
            Sconj_S = torch.sum(sigma[i][:, :, None, None] * torch.conj(psf_ft[i]) * psf_ft[i], dim=1)
            Sconj_I = torch.sum(sigma[i][:, :, None, None] * torch.conj(psf_ft[i]) * images_ft[i], dim=1)
            
            # Use Lofdahl & Scharmer (1994) filter
            if (self.image_filter[i] == 'scharmer'):

                mask = self.lofdahl_scharmer_filter(Sconj_S, Sconj_I)

                out_ft[i] = Sconj_I / Sconj_S
                            
                out_filter_ft[i] = out_ft[i] * mask
            
            # Use simple Wiener filter with tophat prior            
            if (self.image_filter[i] == 'tophat'):
                out_ft[i] = Sconj_I / (Sconj_S + 1e-4)
                out_filter_ft[i] = out_ft[i] * self.mask_diffraction_th[i][None, :, :]

            out_filter[i] = torch.fft.ifft2(out_filter_ft[i]).real
        
        return out_ft, out_filter_ft, out_filter

    
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
        out = image_ft * self.mask_diffraction_th[None, :, :, :]
        return out
            
    def add_frames(self, frames, sigma=None, id_object=0, id_diversity=0, diversity=0.0):
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
        
        self.logger.info(f"Adding frames for object {id_object} - diversity {id_diversity} - defocus {diversity}")

        if sigma is None:
            self.logger.info(f"Estimating noise...")
            sigma = noise.compute_noise(frames)            
            sigma = torch.tensor(sigma.astype('float32')).to(self.device)

        self.ind_object.append(id_object)        
        self.ind_diversity.append(id_diversity)        

        self.frames.append(frames)
        self.sigma.append(sigma)
        self.diversity.append(diversity)

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
        diversity = [None] * self.n_o
        sigma = [None] * self.n_o
        index_frames_diversity = [None] * self.n_o

        # Count the number of frames per object, taking into account the diversity channels
        n_frames_per_object = [0] * self.n_o
        n_diversity_per_object = [0] * self.n_o
        for i in range(self.n_bursts):
            n_frames_per_object[self.ind_object[i]] += n_f
            n_diversity_per_object[self.ind_object[i]] += 1
        
        for i in range(self.n_o):
            frames[i] = torch.zeros(n_seq, n_frames_per_object[i], n_x, n_y)
            diversity[i] = torch.zeros(n_frames_per_object[i])
            sigma[i] = torch.zeros(n_seq, n_frames_per_object[i])
            index_frames_diversity[i] = [0] * n_diversity_per_object[i]

        for i in range(self.n_bursts):

            i_obj = self.ind_object[i]
            i_div = self.ind_diversity[i]

            f0 = i_div * n_f
            f1 = (i_div + 1) * n_f

            index_frames_diversity[i_obj][i_div] = f0

            frames[i_obj][:, f0:f1, :, :] = util.apodize(self.frames[i], self.window)

            diversity[i_obj][f0:f1] = self.diversity[i]
            
            sigma[i_obj][:, f0:f1] = self.sigma[i]
                    
        return frames, diversity, index_frames_diversity, sigma
            
    def deconvolve(self,                                    
                   simultaneous_sequences=1, 
                   infer_object=False, 
                   optimizer='first', 
                   obj_in=None, 
                   modes_in=None,                    
                   n_iterations=20):
        

        """
        Perform deconvolution on a set of frames using specified parameters.
        Parameters:
        -----------
        frames : torch.Tensor
            The input frames to be deconvolved (n_sequences, n_objects, n_frames, nx, ny).
        sigma : torch.Tensor
            The noise standard deviation for each object.
        simultaneous_sequences : int, optional
            Number of sequences to be processed simultaneously (default is 1).
        infer_object : bool, optional
            Whether to infer the object during optimization (default is False).
        optimizer : str, optional
            The optimizer to use ('first' for Adam, 'second' for LBFGS) (default is 'first').
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

        _, self.n_f, self.n_x, self.n_y = self.frames[0].shape

        self.logger.info(f" *****************************************")
        self.logger.info(f" *** SPATIALLY INVARIANT DECONVOLUTION ***")
        self.logger.info(f" *****************************************")

        # Combine all frames
        self.frames_apodized, self.diversity, self.init_frame_diversity, self.sigma = self.combine_frames()

        # Define all basis
        self.define_basis()
        
        # Fill the list of frames and apodize the frames if needed
        for i in range(self.n_o):
            self.frames_apodized[i] = self.frames_apodized[i].to(self.device)
            self.diversity[i] = self.diversity[i].to(self.device)
            self.sigma[i] = self.sigma[i].to(self.device)
            
        self.logger.info(f"Frames")        
        for i in range(self.n_o):
            n_seq, n_f, n_x, n_y = self.frames_apodized[i].shape
            self.logger.info(f"  * Object {i}")
            self.logger.info(f"     - Number of sequences {n_seq}...")
            self.logger.info(f"     - Number of frames {n_f}...")
            self.logger.info(f"     - Number of diversity channels {len(self.init_frame_diversity[i])}...")
            for j, ind in enumerate(self.init_frame_diversity[i]):
                self.logger.info(f"       -> Diversity {j} = {self.diversity[i][ind]}...")
            self.logger.info(f"     - Size of frames {n_x} x {n_y}...")
                
        self.finite_difference = util.FiniteDifference().to(self.device)
        self.set_regularizations()
                                                    
        # Compute the diffraction masks
        self.compute_diffraction_masks()
        
        # Annealing schedules        
        modes = np.cumsum(np.arange(2, self.noll_max+1))
        self.anneal = self.compute_annealing(modes, n_iterations)
                
        # If the regularization parameter is a scalar, we assume that it is the same for all objects
        for reg in self.regularization:
            if reg.type == 'iuwt':                
                if not isinstance(reg.lambda_reg, list):
                    reg.lambda_reg = [reg.lambda_reg] * self.n_o

        #--------------------------------
        # Start optimization
        #--------------------------------

        # Split sequences in batches
        ind = np.arange(n_seq)

        n_seq_total = n_seq

        # Split the sequences in groups of simultaneous sequences to be computed in parallel
        ind = np.array_split(ind, np.ceil(n_seq / simultaneous_sequences))

        n_sequences = len(ind)
        
        self.modes = [None] * n_sequences
        self.loss = [None] * n_sequences

        self.psf_seq = [None] * n_sequences        
        self.degraded_seq = [None] * n_sequences
        self.obj_seq = [None] * n_sequences
        self.obj_diffraction_seq = [None] * n_sequences
        
        for i_seq, seq in enumerate(ind):

            if len(seq) > 1:
                self.logger.info(f"Processing sequences [{seq[0]+1},{seq[-1]+1}]/{n_seq_total}")
            else:
                self.logger.info(f"Processing sequence {seq[0]+1}/{n_seq_total}")

            frames_apodized_seq = []
            frames_ft = []
            sigma_seq = []
            for i in range(self.n_o):
                frames_apodized_seq.append(self.frames_apodized[i][seq, ...])
                frames_ft.append(torch.fft.fft2(self.frames_apodized[i][seq, ...]))
                sigma_seq.append(self.sigma[i][seq, ...])

            n_seq = len(seq)
                
            # frames_apodized_seq = frames_apodized[seq, ...]
            # self.sigma = sigma[seq, ...]
            # self.weight = 1.0 / sigma[seq, ...]
                                
            if (infer_object):

                # Find frame with best contrast
                contrast = torch.std(frames_apodized_seq, dim=(-1, -2)) / torch.mean(frames_apodized_seq, dim=(-1, -2)) * 100.0        
                ind = torch.argsort(contrast[0, 0, :], descending=True)

                if obj_in is not None:
                    self.logger.info(f"Using provided initial object...")
                    obj_init = obj_in
                    obj_init = obj_init.to(self.device)
                else:
                    if self.config['initialization']['object'] == 'contrast':
                        self.logger.info(f"Selecting initial object as image with best contrast...")
                        obj_init = frames_apodized_seq[:, :, ind[0], :, :]
                    if self.config['initialization']['object'] == 'average':
                        self.logger.info(f"Selecting initial object as average image...")
                        obj_init = torch.mean(frames_apodized_seq, dim=2)
                
                # Initialize the object with the inverse softplus
                if (self.config['optimization']['transform'] == 'softplus'):
                    obj_init = torch.log(torch.exp(obj_init) - 1.0)
                                
            # Unknown modes
            if modes_in is not None:
                self.logger.info(f"Using provided initial modes...")
                modes = modes_in.clone().detach().to(self.device).requires_grad_(True)
            else:
                if self.config['initialization']['modes_std'] == 0:
                    self.logger.info(f"Initializing modes with zeros...")
                    modes = torch.zeros((n_seq, self.n_f, self.n_modes), device=self.device, requires_grad=True)
                else:
                    self.logger.info(f"Initializing modes with random values with standard deviation {self.config['initialization']['modes_std']}")
                    modes = self.config['initialization']['modes_std'] * torch.randn((n_seq, self.n_f, self.n_modes))
                    modes = modes.clone().detach().to(self.device).requires_grad_(True)            

            # Second order optimizer
            if optimizer == 'second':
                if (infer_object):
                    obj = obj_init.clone().detach().to(self.device).requires_grad_(True)
                    parameters = [modes, obj]
                else:                
                    parameters = [modes]

                opt = torch.optim.LBFGS(parameters, lr=0.01)

            else:

                if (infer_object):
                    self.logger.info(f"Optimizing object and modes...")
                    obj = obj_init.clone().detach().to(self.device).requires_grad_(True)
                    parameters = [{'params': modes, 'lr': self.lr_modes}, {'params': obj, 'lr': self.lr_obj}]
                else:
                    self.logger.info(f"Optimizing modes only...")                
                    parameters = [{'params': modes, 'lr': self.lr_modes}]

                opt = torch.optim.Adam(parameters)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3*n_iterations)

            losses = torch.zeros(n_iterations, device=self.device)
            contrasts = torch.zeros(n_iterations, device=self.device)

            t = tqdm(range(n_iterations))

            _, self.psf_diffraction_ft = self.compute_psf_diffraction()

            n_active = 2
                
            for loop in t:
            
                def closure():
                    opt.zero_grad(set_to_none=True)
                
                    # Compute PSF from current wavefront coefficients and reference 
                    modes_centered = modes.clone()
                    modes_centered[:, :, 0:2] = modes_centered[:, :, 0:2] - modes[:, 0:1, 0:2]
                                
                    # modes -> (n_seq, n_f, self.n_modes)
                    psf, psf_ft = self.compute_psfs(modes_centered[:, :, 0:n_active])
                    
                    if (infer_object):
                        
                        # Compute filtered object from the current estimate while also clamping negative values
                        if (self.config['optimization']['transform'] == 'softplus'):
                            tmp = torch.clamp(F.softplus(obj), min=0.0)
                            obj_ft = torch.fft.fft2(tmp)
                        else:
                            tmp = torch.clamp(obj, min=0.0)
                            obj_ft = torch.fft.fft2(obj)

                        # Filter in Fourier
                        obj_filter_ft = self.fft_filter(obj_ft)                        

                        degraded_ft = obj_ft[:, :, None, :, :] * psf_ft

                        residual = self.weight[:, :, None, None, None] * (degraded_ft - frames_ft)
                        loss_mse = torch.mean((residual * torch.conj(residual)).real) / self.npix**2

                    else:                        
                        obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, psf_ft, sigma_seq)

                        loss_mse = torch.tensor(0.0).to(self.device)

                        # Sum over objects, frames and diversity channels
                        for i in range(self.n_o):
                            Q = torch.mean(self.sigma[i]) + torch.sum(psf_ft[i] * torch.conj(psf_ft[i]), dim=1)
                            t1 = torch.sum(frames_ft[i] * torch.conj(frames_ft[i]), dim=1)
                            t2 = torch.sum(torch.conj(frames_ft[i]) * psf_ft[i], dim=1)                        
                            loss_mse += torch.mean(t1 - t2 * torch.conj(t2) / Q).real / self.npix**2
                                            
                                        
                    # Object regularization
                    loss_obj = torch.tensor(0.0).to(self.device)
                    for index in self.index_regularization['object']:
                        loss_obj += self.regularization[index](obj_filter)                        
                    
                    # Total loss
                    loss = loss_mse + loss_obj
                                                        
                    # Save some information for the progress bar
                    self.loss_local = loss.detach()
                    self.obj_filter = [None] * self.n_o
                    for i in range(self.n_o):
                        self.obj_filter[i] = obj_filter[i].detach()
                    self.loss_mse_local = loss_mse.detach()
                    self.loss_obj_local = loss_obj.detach()

                    loss.backward()

                    return loss
                        
                opt.step(closure)

                scheduler.step()

                if self.cuda:
                    gpu_usage = f'{self.handle.gpu_utilization()}'            
                    memory_usage = f'{self.handle.memory_used() / 1024**2:4.1f}/{self.handle.memory_total() / 1024**2:4.1f} MB'
                    memory_pct = f'{self.handle.memory_used() / self.handle.memory_total() * 100.0:4.1f}%'
                                   
                tmp = OrderedDict()                
                
                if self.cuda:
                    tmp['gpu'] = f'{gpu_usage} %'
                    tmp['mem'] = f'{memory_usage} ({memory_pct})'

                tmp['active'] = f'{n_active}'
                tmp['contrast'] = f'{torch.std(self.obj_filter[0]) / torch.mean(self.obj_filter[0]) * 100.0:7.4f}'
                tmp['minmax'] = f'{torch.min(self.obj_filter[0]):7.4f}/{torch.max(self.obj_filter[0]):7.4f}'
                tmp['LMSE'] = f'{self.loss_mse_local.item():8.6f}'                
                tmp['LOBJ'] = f'{self.loss_obj_local.item():8.6f}'
                tmp['L'] = f'{self.loss_local.item():7.4f}'
                t.set_postfix(ordered_dict=tmp)

                n_active = self.anneal[loop]    
                
            modes_centered = modes.clone().detach()
            modes_centered[:, :, 0:2] = modes_centered[:, :, 0:2] - modes_centered[:, 0:1, 0:2]
            psf, psf_ft = self.compute_psfs(modes_centered)

            if (infer_object):

                # Compute filtered object from the current estimate
                if (self.config['optimization']['transform'] == 'softplus'):
                    obj_ft = torch.fft.fft2(F.softplus(obj))
                else:
                    obj_ft = torch.fft.fft2(obj)

                # Filter in Fourier
                obj_filter_ft = self.fft_filter(obj_ft)                

            else:
                obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, psf_ft, sigma_seq)  
                                   
            obj_filter_diffraction = [None] * self.n_o
            degraded = [None] * self.n_o
            for i in range(self.n_o):                
                obj_filter_diffraction[i] = torch.fft.ifft2(obj_filter_ft[i] * self.psf_diffraction_ft[i][None, :, :]).real
            
                # Compute final degraded images
                degraded_ft = obj_filter_ft[i][:, None, :, :] * psf_ft[i]
                degraded[i] = torch.fft.ifft2(degraded_ft).real
            
            # Store the results for the current set of sequences
            self.modes[i_seq] = modes.detach()
            self.loss[i_seq] = losses.detach()

            for i in range(self.n_o):
                psf[i] = psf[i].detach()
                degraded[i] = degraded[i].detach()
                obj_filter[i] = obj_filter[i].detach()
                obj_filter_diffraction[i] = obj_filter_diffraction[i].detach()

            self.psf_seq[i_seq] = psf
            self.degraded_seq[i_seq] = degraded
            self.obj_seq[i_seq] = obj_filter
            self.obj_diffraction_seq[i_seq] = obj_filter_diffraction        
        
        # Concatenate the results from all sequences and all objects independently
        self.psf = [None] * self.n_o
        self.degraded = [None] * self.n_o
        self.obj = [None] * self.n_o
        self.obj_diffraction = [None] * self.n_o

        # for i in range(self.n_o):
        self.modes = torch.cat(self.modes, dim=0)
        self.loss = torch.cat(self.loss, dim=0)
        
        
        for i in range(self.n_o):
            tmp = [self.psf_seq[j][i] for j in range(n_sequences)]
            self.psf[i] = torch.cat(tmp, dim=0)

            tmp = [self.degraded_seq[j][i] for j in range(n_sequences)]
            self.degraded[i] = torch.cat(tmp, dim=0)

            tmp = [self.obj_seq[j][i] for j in range(n_sequences)]
            self.obj[i] = torch.cat(tmp, dim=0)

            tmp = [self.obj_diffraction_seq[j][i] for j in range(n_sequences)]
            self.obj_diffraction[i] = torch.cat(tmp, dim=0)
        
        return 
    
if __name__ == '__main__':
    pass