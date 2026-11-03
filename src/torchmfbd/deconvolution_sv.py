import warnings
import numpy as np
import scipy.optimize as opt
import torchmfbd.util as util
import torch
import torch.nn.functional as F
from torchmfbd.deconvolution import Deconvolution
from torchmfbd.warp import warp
from tqdm import tqdm
from einops import rearrange
from collections import OrderedDict

class DeconvolutionSV(Deconvolution):
    def __init__(self, config):
                
        super().__init__(config)
        
        # self.ngrid_modes = config['ngrid_modes']
        self.psf_model = self.config['psf']['model']

        # Model first: zernike/kl/vae configurations have no psf.filename
        if self.psf_model.lower() not in ['pca', 'nmf']:
            raise Exception(f"Spatially variant deconvolution cannot be used with PSF model {self.psf_model}")

        # Settings of the spatially invariant class that this class does not use: said, not refused,
        # because shared configuration files carry them
        ignored = []
        if self.loss_type != 'joint':
            ignored.append(f"optimization.loss_type = {self.loss_type} (the loss is a weighted MSE)")
        if self.config['optimization'].get('filter_loss', 'wiener') != 'wiener':
            ignored.append(f"optimization.filter_loss = {self.config['optimization']['filter_loss']}")
        if self.use_jitter != 'none':
            ignored.append(f"psf.jitter = {self.use_jitter}")
        for item in ignored:
            msg = f"DeconvolutionSV does not use {item}; the setting is ignored"
            self.logger.warning(msg)
            warnings.warn(msg, UserWarning)

        self.basis_file = self.config['psf']['filename']
        self.ngrid_modes = self.config['psf']['ngrid_modes']

        # PSF model parameterized with the PSF
        if (self.psf_model.lower() in ['pca', 'nmf']):

            self.logger.info(f"PSF model: linear direct expansion")

            if (self.psf_model.lower() not in ['pca', 'nmf']):
                raise Exception(f"Unknown basis {self.basis}")
            
            # PSF modes       
            self.logger.info(f"  * Using {self.psf_model.lower()} modes")
            self.logger.info(f"     - Reading PSF modes from {self.basis_file}...")
            f = np.load(self.basis_file)

            # Extract wavelength, diameter, pixel size, and central obscuration from the file with the modes
            if self.config['images']['wavelength'] != f['info'][0]:
                raise Exception(f"Wavelength mismatch: {self.config['images']['wavelength']} vs {f['info'][0]}")
            if self.config['telescope']['diameter'] != f['info'][1]:
                raise Exception(f"Diameter mismatch: {self.config['telescope']['diameter']} vs {f['info'][1]}")
            if self.config['images']['pix_size'] != f['info'][2]:
                raise Exception(f"Pixel size mismatch: {self.config['images']['pix_size']} vs {f['info'][2]}")
            if self.config['telescope']['central_obscuration'] != f['info'][3]:
                raise Exception(f"Central obscuration mismatch: {self.config['telescope']['central_obscuration']} vs {f['info'][3]}")

            n_psf = int(np.sqrt(f['basis'].shape[1]))
            if f['basis'].shape[0] < self.n_modes:
                raise ValueError(f"The basis file has {f['basis'].shape[0]} modes but psf.nmax_modes requests {self.n_modes}")
            self.logger.info(f"  * Size of basis PSF modes {n_psf}x{n_psf}...")        
            self.logger.info(f"  * Number of modes {self.n_modes}...")            
            
            self.psf_modes = f['basis'][0:self.n_modes, :].reshape((self.n_modes, n_psf, n_psf))
            self.psf_diffraction = f['psf_diffraction']        

            # Unified resize strategy: Pad or crop PSF modes to the size of the image BEFORE normalization
            if n_psf > self.npix:
                self.psf_modes = self.psf_modes[:, n_psf//2 - self.npix//2 : n_psf//2 + self.npix//2, n_psf//2 - self.npix//2 : n_psf//2 + self.npix//2]
                self.psf_diffraction = self.psf_diffraction[n_psf//2 - self.npix//2 : n_psf//2 + self.npix//2, n_psf//2 - self.npix//2 : n_psf//2 + self.npix//2]
            elif n_psf < self.npix:
                pad = (self.npix - n_psf) // 2
                self.psf_modes = np.pad(self.psf_modes, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
                self.psf_diffraction = np.pad(self.psf_diffraction, ((pad, pad), (pad, pad)), mode='constant')

            if (self.psf_model.lower() == 'pca'):
                self.logger.info(f"Computing PCA coefficients of diffraction PSF")
                self.coeffs_diffraction = np.sum(self.psf_modes * self.psf_diffraction[None, ...], axis=(1, 2))
                self.modes_init = np.copy(self.coeffs_diffraction)

            if (self.psf_model.lower() == 'nmf'):
                # 'Fat wing' exponent, as in the spatially invariant class
                basis_power = self.config['psf'].get('basis_power', 1.0)
                if basis_power != 1.0:
                    self.logger.info(f"  * Applying 'fat wing' power: {basis_power}")
                    self.psf_modes = (self.psf_modes + 1e-12)**basis_power

                # Normalize the modes to unit area (an all-zero mode stays zero)
                self.psf_modes /= (np.sum(self.psf_modes, axis=(1, 2), keepdims=True) + 1e-12)

                self.logger.info(f"Computing NMF coefficients of diffraction PSF")            
                                        
                self.coeffs_diffraction, _ = opt.nnls(self.psf_modes.reshape((self.n_modes, self.npix**2)).T, self.psf_diffraction.flatten())
                
                # Since the diffraction PSF is already normalized to unit area, the sum of the inferred coefficients must be 1
                # We renormalize the coefficients to ensure that
                self.coeffs_diffraction /= np.sum(self.coeffs_diffraction)
                            
                min_coeff = np.min(self.coeffs_diffraction[self.coeffs_diffraction > 0])
                ind = np.where(self.coeffs_diffraction == 0)[0]            
                self.modes_init = np.copy(self.coeffs_diffraction)
                self.modes_init[ind] = 0.1 * min_coeff * np.random.rand(len(ind))            
        else:
            raise Exception(f"Spatially variant deconvolution cannot be used with PSF model {self.psf_model}")
                          
                    
        if (self.ngrid_modes != self.npix):
            self.logger.info(f"  * Using tiptilt and modes of size {self.ngrid_modes}x{self.ngrid_modes} and bilinear interpolation")
        else:
            self.logger.info(f"  * Using tiptilt and modes of size {self.ngrid_modes}x{self.ngrid_modes}")        

        # Shift the modes and diffraction PSF to later compute FFTs
        self.psf_modes = np.fft.ifftshift(self.psf_modes, axes=(1, 2))
        self.psf_diffraction = np.fft.ifftshift(self.psf_diffraction) 
                                                            
        # Move all relevant quantities to GPU
        self.psf_modes = torch.tensor(self.psf_modes.astype('float32')).to(self.device)
        self.psf_modes_fft = torch.fft.rfft2(self.psf_modes)                            
        self.modes_init = torch.tensor(self.modes_init.astype('float32'))

        # Learning rates
        self.lr_tiptilt = self.config['optimization']['lr_tt']
        self.logger.info(f"Optimization")
        self.logger.info(f"  * lr_obj  ={self.lr_obj}")
        self.logger.info(f"  * lr_tt   ={self.lr_tiptilt}")
        self.logger.info(f"  * lr_modes={self.lr_modes}")
        self.logger.info(f"  * Scale of softplus {self.config['optimization']['softplus_scale']}...")
        
    def define_basis(self):

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

        # Now iterate over all unique wavelengths and associate the basis
        # to the corresponding object
        for i in range(len(unique_wavelengths)):

            wavelength = unique_wavelengths[i]
                    
            # Compute the diffraction limit and the frequency grid
            cutoff = self.config['telescope']['diameter'] / (wavelength * 1e-8) / 206265.0
            freq = np.fft.fftfreq(self.npix, d=self.config['images']['pix_size']) / cutoff
            
            xx, yy = np.meshgrid(freq, freq)
            rho = np.sqrt(xx ** 2 + yy ** 2).astype('float32')
            rho = torch.tensor(rho).to(self.device)

            diffraction_limit = wavelength * 1e-8 / self.config['telescope']['diameter'] * 206265.0

            self.logger.info(f"Wavelength {i} ({wavelength} A)")
            self.logger.info(f"  * Diffraction: {diffraction_limit} arcsec")
            self.logger.info(f"  * Diffraction (x1.22): {1.22 * diffraction_limit} arcsec")

            for j in range(self.n_o):
                if ind_wavelengths[j] == i:
                    self.rho[j] = rho
                    self.diffraction_limit[j] = diffraction_limit                

        return

    def fft_filter_image(self, obj, mask_diffraction):
        """
        Filter the object in Fourier space
        It simply multiplies the FFT of the object (properly apodized) by the Fourier filter
        and returns the inverse FFT of the result. This can be used to avoid structures above
        the diffraction limit.
        """
        
        # Apodize the object
        
        mean_val = torch.mean(obj, dim=(-1, -2), keepdims=True)
        obj1 = obj - mean_val        
        obj2 = obj1 * self.window[None, :, :]
        obj3 = obj2 + mean_val

        del obj1, obj2

        # Apply the mask
        objf1 = torch.fft.fft2(obj3)        
        objf2 = objf1 * mask_diffraction[None, ...]
        obj = torch.fft.ifft2(objf2).real

        del objf1, objf2

        return obj
    
    def compute_syn(self, ns, nf, obj_filtered, tiptilt, modes, infer_tiptilt, infer_modes, i_o):
        """
        Compute the synthetic image based on the inferred object, tip-tilt, and modes.
        Receives tiptilt and modes already interpolated to full (npix x npix) resolution.

        Parameters:
        im (torch.Tensor): Input image tensor of shape (ns, no, nf, nx, ny).
        obj_filtered (torch.Tensor): Filtered object tensor.
        tiptilt (torch.Tensor): Tip-tilt tensor, already at npix resolution.
        modes (torch.Tensor): Modes tensor, already at npix resolution.
        infer_tiptilt (bool): Whether tip-tilt is being inferred.
        infer_modes (bool): Flag indicating whether to infer modes or simply apply tip-tilt.

        Returns:
        torch.Tensor: Synthetic image tensor.
        """

        nx, ny = self.npix, self.npix
                
        # Apply tip-tilt to the object by applying a warp with the pixel-dependent optical flow
        if infer_tiptilt:
            tmp_obj = obj_filtered[:, None, None, :, :].expand(ns, nf, 1, self.npix, self.npix)
            tmp_obj = rearrange(tmp_obj, 'b nf no nx ny -> (b nf) no nx ny')
            tmp_tt = rearrange(tiptilt, 'b nf d nx ny -> (b nf) d nx ny')
            obj_tt = warp(tmp_obj, tmp_tt)
            obj_tt = rearrange(obj_tt, '(b nf) no nx ny -> b no nf nx ny', b=ns)

            del tmp_obj, tmp_tt
        else:
            obj_tt = obj_filtered[:, None, None, :, :].expand(ns, 1, nf, self.npix, self.npix)
            
        # Now apply the effect of the high order modes
        if infer_modes:
            
            # If we are using NMF, we need to force the coefficients to be non-negative
            # and also normalize by their sum to force the PSF to have unit area
            if self.psf_model.lower() == 'nmf':                
                scale = self.config['optimization'].get('softplus_scale', 1.0)
                modes_nn = F.softplus(scale * modes) / scale
                                
                modes_unitarea = modes_nn / (torch.sum(modes_nn, dim=2, keepdims=True) + 1e-12)
                
                del modes_nn
            else:
                modes_unitarea = modes
            
            # Optimization: Apply apodization window to the object ONCE before mode expansion
            # This is equivalent to applying it per-mode but much faster
            mn = torch.mean(obj_tt, dim=(-1, -2), keepdims=True)
            obj_tt_apod = (obj_tt - mn) * self.window[None, None, None, :, :] + mn
            
            # Compute the product of the apodized object and the modes
            # shape: (ns, nf, 1, n_modes, nx, ny) * (ns, 1, nf, n_modes, nx, ny) -> 6D
            obj_tt_expanded = obj_tt_apod[:, :, :, None, :, :] * modes_unitarea[:, None, :, :, :, :]
            
            # Compute convolution of current object with the PSF modes using rfft2
            obj_tt_modes_fft = torch.fft.rfft2(obj_tt_expanded)
            
            # Optimization: Sum in Fourier domain using einsum to avoid 6D memory allocation
            # By linearity: sum_m IFFT(X_m * H_m) == IFFT(sum_m X_m * H_m)
            # syn_ft shape: (ns, no, nf, freq_x, freq_y_half)
            syn_ft = torch.einsum('b...mxy, mxy -> b...xy', obj_tt_modes_fft, self.psf_modes_fft)
            syn = torch.fft.irfft2(syn_ft, s=(self.npix, self.npix))

            del obj_tt_modes_fft, obj_tt_expanded, syn_ft

            # In the case of the PCA basis, we need to normalize by the
            # convolution of the modes with the PCA basis
            if self.psf_model.lower() == 'pca':
                mn = torch.mean(modes, dim=(-1, -2), keepdims=True)
                modes_norm = (modes - mn) * self.window[None, None, None, :, :] + mn
                
                # Compute convolution using rfft2, summing in Fourier domain using einsum
                modes_norm_fft = torch.fft.rfft2(modes_norm)
                norm_ft = torch.einsum('b...mxy, mxy -> b...xy', modes_norm_fft, self.psf_modes_fft)
                norm = torch.fft.irfft2(norm_ft, s=(self.npix, self.npix))
                
                syn = syn / norm

                del norm, norm_ft, modes_norm_fft, modes_norm

        else:
            syn = obj_tt
            modes_unitarea = None
        
        return syn[:, 0, ...], tiptilt, modes_unitarea
                            
    def deconvolve(self,                   
                   simultaneous_sequences=1,
                   infer_tiptilt=True, 
                   infer_modes=True, 
                   tiptilt_init=None,
                   n_iterations=20,
                   batch_size=64):
        
        """
        Perform spatially variant deconvolution on a set of frames.
        
        Parameters:
        -----------
        simultaneous_sequences : int, optional
            The number of sequences to process simultaneously. Default is 1.
        infer_tiptilt : bool, optional
            Whether to infer tip-tilt. Default is True.
        infer_modes : bool, optional
            Whether to infer modes. Default is True.
        tiptilt_init : torch.Tensor, optional
            Initial tip-tilt values. Default is None.
        n_iterations : int, optional
            Number of iterations for the optimization. Default is 20.
        batch_size : int, optional
            Batch size for processing frames. Default is 64.
        
        Returns:
        --------
        obj : numpy.ndarray
            The deconvolved object.
        tiptilt_infer : numpy.ndarray
            The inferred tip-tilt values.
        tiptilt : numpy.ndarray
            The final tip-tilt values.
        modes_infer : numpy.ndarray
            The inferred modes.
        modes : numpy.ndarray
            The final modes.
        coeffs_diffraction : numpy.ndarray
            The coefficients of diffraction.
        losses : list
            The list of loss values during optimization.
        psf_modes : numpy.ndarray
            The point spread function modes.
        """
        
        _, self.n_f, self.n_x, self.n_y = self.frames[0].shape

        self.logger.info(f" ***************************************")
        self.logger.info(f" *** SPATIALLY VARIANT DECONVOLUTION ***")
        self.logger.info(f" ***************************************")

        # Combine all frames
        self.frames_apodized, self.diversity, self.init_frame_diversity, self.sigma, self.plane = self.combine_frames()

        # Precompute the FFT of the frames to speed up potential object updates
        self.frames_ft = [None] * self.n_o
        for i in range(self.n_o):
            self.frames_ft[i] = torch.fft.fft2(self.frames_apodized[i], norm=self.fft_norm)

        # Define all basis
        self.define_basis()

        # The PSF basis has no wavefront to add a defocus to, and the loop fits n_f frames per object
        for i in range(self.n_o):
            if len(self.init_frame_diversity[i]) > 1:
                raise ValueError(f"Phase diversity ({len(self.init_frame_diversity[i])} channels for object {i}) is not available in DeconvolutionSV")

        # Fill the list of frames and apodize the frames if needed
        for i in range(self.n_o):
            self.frames_apodized[i] = self.frames_apodized[i].to(self.device)
            self.diversity[i] = self.diversity[i].to(self.device)
            self.sigma[i] = self.sigma[i].to(self.device)
            self.frames_ft[i] = self.frames_ft[i].to(self.device)
            if self.remove_gradient_apodization:
                self.plane[i] = self.plane[i].to(self.device)
            
        self.logger.info(f"Frames")        
        for i in range(self.n_o):
            n_seq, n_f, n_x, n_y = self.frames_apodized[i].shape
            self.logger.info(f"  * Object {i}")            
            self.logger.info(f"     - Number of sequences {n_seq}...")
            self.logger.info(f"     - Number of frames {n_f}...")
            self.logger.info(f"     - Number of diversity channels {len(self.diversity[i])}...")
            for j, ind in enumerate(self.init_frame_diversity[i]):
                self.logger.info(f"       -> Diversity {j} = {self.diversity[i][ind]}...")
            self.logger.info(f"     - Size of frames {n_x} x {n_y}...")
                
        self.finite_difference = util.FiniteDifference().to(self.device)
        self.set_regularizations()

        # Compute the diffraction masks
        self.compute_diffraction_masks()
        
        # Annealing schedules — values are mode counts (same convention as SI NMF path)
        n_anneal = max(2, (self.n_modes - 2) // 5)
        modes = np.linspace(2, self.n_modes, n_anneal).astype(int)
        self.anneal = self.compute_annealing(modes, n_iterations)

        # If the regularization parameter is a scalar, we assume that it is the same for all objects
        for reg in self.regularization:
            if reg.type == 'iuwt':                
                if not isinstance(reg.lambda_reg, list):
                    reg.lambda_reg = [reg.lambda_reg] * self.n_o


        self.batch_size = batch_size
        
        # self.sigma = sigma
        # self.weight = 1.0 / sigma
      
        
        #--------------------------------
        # Start optimization
        #--------------------------------

        # Split sequences in batches
        ind = np.arange(n_seq)

        n_seq_total = n_seq

        # Split the sequences in groups of simultaneous sequences to be computed in parallel
        ind = np.array_split(ind, np.ceil(n_seq / simultaneous_sequences))        

        n_sequences = len(ind)

        self.modes_lr_seq = [None] * n_sequences
        self.modes_hr_seq = [None] * n_sequences
        self.tiptilt_lr_seq = [None] * n_sequences
        self.tiptilt_hr_seq = [None] * n_sequences
        self.loss = [None] * n_sequences        
        self.obj_seq = [None] * n_sequences
                
        for i_seq, seq in enumerate(ind):

            if len(seq) > 1:
                self.logger.info(f"Processing sequences [{seq[0]+1},{seq[-1]+1}]/{n_seq_total}")
            else:
                self.logger.info(f"Processing sequence {seq[0]+1}/{n_seq_total}")

            frames_apodized_seq = []
            frames_ft = []
            sigma_seq = []
            weight_seq = []
            for i in range(self.n_o):
                frames_apodized_seq.append(self.frames_apodized[i][seq, ...])
                frames_ft.append(self.frames_ft[i][seq, ...])
                sigma_seq.append(self.sigma[i][seq, ...])
                s2_mean = torch.mean(self.sigma[i][seq, ...]**2, dim=1, keepdim=True)
                weight_seq.append(s2_mean / (self.sigma[i][seq, ...]**2 + 1e-10))

            # Sequences in this batch: tip-tilt and modes are allocated per batch
            n_seq = len(seq)
                        
            # Initial estimation of the object: average of all frames
            if self.config['initialization']['object'].lower() == 'average':
                self.logger.info(f"Using the mean of the frames as initialization")
                obj = []
                for i in range(self.n_o):
                    obj.append(torch.mean(frames_apodized_seq[i], dim=1))
                    if not self.remove_gradient_apodization:
                        obj[i] /= torch.mean(obj[i])
                
            # Select the frame with the highest contrast
            if self.config['initialization']['object'].lower() == 'contrast':
                self.logger.info(f"Using the frame with highest contrast as initialization")

                obj = []
                for i in range(self.n_o):
                    if self.remove_gradient_apodization:
                        contrast = torch.std(frames_apodized_seq[i], dim=(-1, -2))
                    else:
                        contrast = torch.std(frames_apodized_seq[i], dim=(-1, -2)) / torch.mean(frames_apodized_seq[i], dim=(-1, -2))
                    
                    best = torch.argmax(contrast, dim=1)
                    obj.append(frames_apodized_seq[i][torch.arange(n_seq), best, ...])
                    if not self.remove_gradient_apodization:
                        obj[i] /= torch.mean(obj[i])
                                            
            
            if infer_tiptilt and infer_modes:
                self.logger.info("Optimizing object, tip-tilt, and modes...")
            if infer_tiptilt and not infer_modes:
                self.logger.info("Optimizing object and tip-tilt...")
                
            # Initialization of the tip-tilt            
            if tiptilt_init is None:
                tiptilt = torch.zeros((n_seq, self.n_f, 2, self.ngrid_modes, self.ngrid_modes))
            else:
                tiptilt = tiptilt_init[seq, ...]
            
            # Initialization of the modes
            # account for softplus_scale: x = inv_softplus(tmp * scale) / scale
            scale = self.config['optimization'].get('softplus_scale', 1.0)
            
            # Safe inverse softplus: for large values x ~= y, for small values use the log formula
            threshold = 20.0
            if self.psf_model.lower() == 'nmf':
                modes_val = torch.where(self.modes_init * scale > threshold, 
                                      self.modes_init, 
                                      torch.log(torch.exp(self.modes_init * scale) - 1.0 + 1e-10) / scale)
            else:
                # PCA coefficients are signed and enter compute_syn untransformed
                modes_val = self.modes_init.clone()
            
            modes = modes_val[None, None, :, None, None].repeat((n_seq, self.n_f, 1, self.ngrid_modes, self.ngrid_modes))
            
            # Variables to be optimized
            obj_infer = []
            for i in range(self.n_o):
                obj_infer.append(obj[i].clone().detach().to(self.device).requires_grad_(True))
            modes_infer = modes.clone().detach().to(self.device).requires_grad_(True)
            tiptilt_infer = tiptilt.clone().detach().to(self.device).requires_grad_(True)
            
            # Parameters to be optimized
            parameters = [{'params': obj_infer, 'lr': self.lr_obj}, {'params': modes_infer, 'lr': self.lr_modes}, {'params': tiptilt_infer, 'lr': self.lr_tiptilt}]
            
            # Define the optimizer and the scheduler
            optimizer = torch.optim.AdamW(parameters)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3*n_iterations)            
            
            self.logger.info("Starting optimization...")

            losses = []
        
            t = tqdm(range(n_iterations))
                    
            for loop in t:
                                    
                optimizer.zero_grad(set_to_none=True)
            
                # Partition the frames in batches
                indices = np.arange(self.n_f)
                
                indices = np.array_split(indices, np.ceil(self.n_f / self.batch_size))               

                # Accumulate the MSE loss for all batches
                loss_mse = torch.tensor(0.0).to(self.device)
                loss_tiptilt = torch.tensor(0.0).to(self.device)
                loss_modes = torch.tensor(0.0).to(self.device)
                loss_obj = torch.tensor(0.0).to(self.device)

                # Interpolate tiptilt and modes once per optimizer step (PERF-SV-A)
                # tiptilt_infer/modes_infer don't change across frame batches within a step.
                if self.ngrid_modes != self.npix:
                    tmp = rearrange(tiptilt_infer, 'b nf d nx ny -> (b nf) d nx ny')
                    tiptilt_hr = F.interpolate(tmp, size=(self.npix, self.npix), mode='bilinear')
                    tiptilt_hr = rearrange(tiptilt_hr, '(b nf) d nx ny -> b nf d nx ny', b=n_seq)
                    del tmp

                    tmp = rearrange(modes_infer, 'b nf nm nx ny -> (b nf) nm nx ny')
                    modes_hr = F.interpolate(tmp, size=(self.npix, self.npix), mode='bilinear')
                    modes_hr = rearrange(modes_hr, '(b nf) nm nx ny -> b nf nm nx ny', b=n_seq)
                    del tmp
                else:
                    tiptilt_hr = tiptilt_infer
                    modes_hr = modes_infer

                obj_filtered = [None] * self.n_o
                for i in range(self.n_o):
                    # Filter the object in the Fourier domain
                    if self.image_filter[i] == 'tophat':                        
                        obj_filtered[i] = self.fft_filter_image(obj_infer[i], self.mask_diffraction_th[i])
                    else:
                        obj_filtered[i] = obj_infer[i]
                    
                for i in range(self.n_o):
                    for j, ind in enumerate(indices):
                                                
                        # Compute the synthetic images
                        syn, tiptilt, modes = self.compute_syn(n_seq, len(ind),
                                                                             obj_filtered[i], 
                                                                             tiptilt_hr[:, ind, ...],
                                                                             modes_hr[:, ind, ...],
                                                                             infer_tiptilt, 
                                                                             infer_modes,
                                                                             i)
                                                
                        # Compute the MSE loss                        
                        loss_mse += torch.mean(weight_seq[i][:, ind, None, None] * (syn - frames_apodized_seq[i][:, ind, :, :])**2)
                
                # Regularization
                # Tip-tilt regularization                
                for index in self.index_regularization['tiptilt']:                    
                    loss_tiptilt += self.regularization[index](tiptilt_infer)

                # Modes regularization                
                for index in self.index_regularization['modes']:
                    loss_modes += self.regularization[index](modes_infer)

                # Object regularization
                for index in self.index_regularization['object']:
                    loss_obj += self.regularization[index](obj_filtered)
                
                # Total loss
                loss = loss_mse + loss_tiptilt + loss_modes + loss_obj

                # Backward pass
                loss.backward()
                
                # Update the parameters
                optimizer.step()
                
                # Update the learning rate
                scheduler.step()
                
                
                # Do some printing
                if self.handle is not None:
                    gpu_usage = f'{self.handle.gpu_utilization()}'            
                    memory_usage = f'{self.handle.memory_used() / 1024**2:4.1f}/{self.handle.memory_total() / 1024**2:4.1f} MB'
                    memory_pct = f'{self.handle.memory_used() / self.handle.memory_total() * 100.0:4.1f}%'
                else:
                    gpu_usage = '0.0'
                    memory_usage = '0.0/0.0 MB'
                    memory_pct = '0.0%'
                
                current_lr_obj = optimizer.param_groups[0]['lr']
                current_lr_modes = optimizer.param_groups[1]['lr']

                tmp = OrderedDict()
                if self.cuda:
                    tmp['gpu'] = f'{gpu_usage} %'
                    tmp['mem'] = f'{memory_usage} ({memory_pct})'
                tmp['lrm'] = f'{current_lr_modes:8.6f}'
                tmp['lro'] = f'{current_lr_obj:8.6f}'                
                tmp['contrast'] = f'{torch.std(obj_filtered[0]) / torch.mean(obj_filtered[0]) * 100.0:7.4f}'
                tmp['minmax'] = f'{torch.min(obj_filtered[0]):7.4f}/{torch.max(obj_filtered[0]):7.4f}'
                tmp['LMSE'] = f'{loss_mse.item():8.6f}'
                tmp['LTT'] = f'{loss_tiptilt.item():8.6f}'
                tmp['LMOD'] = f'{loss_modes.item():8.6f}'
                tmp['LOBJ'] = f'{loss_obj.item():8.6f}'
                tmp['L'] = f'{loss.item():8.6f}'

                t.set_postfix(ordered_dict = tmp)

                losses.append(loss.detach())

            self.loss[i_seq] = torch.stack(losses)

            if self.psf_model.lower() == 'nmf':
                scale = self.config['optimization'].get('softplus_scale', 1.0)
                modes_infer = F.softplus(scale * modes_infer) / scale
                # modes_infer = F.relu(modes_infer, inplace=False)
                modes_infer /= (torch.sum(modes_infer, dim=2, keepdims=True) + 1e-12)

            # Final interpolation of the tiptilt and modes to the appropriate sizes
            if (self.ngrid_modes != self.npix):                
                tmp = rearrange(tiptilt_infer, 'b nf d nx ny -> (b nf) d nx ny')
                tiptilt = F.interpolate(tmp, size=(self.npix, self.npix), mode='bilinear')
                tiptilt = rearrange(tiptilt, '(b nf) d nx ny -> b nf d nx ny', b=n_seq)

                del tmp

                tmp = rearrange(modes_infer, 'b nf nm nx ny -> (b nf) nm nx ny')
                modes = F.interpolate(tmp, size=(self.npix, self.npix), mode='bilinear')
                modes = rearrange(modes, '(b nf) nm nx ny -> b nf nm nx ny', b=n_seq)

                del tmp
            else:
                tiptilt = tiptilt_infer
                modes = modes_infer

            # Add back the gradient plane that was subtracted during apodization
            # Mirrors base class compute_object: out_filter[i] += plane[i][:, 0, :, :]
            if self.remove_gradient_apodization:
                for i in range(self.n_o):
                    obj_filtered[i] = obj_filtered[i] + self.plane[i][seq, 0, :, :]

            # Denormalize the object and the frames and move to the CPU
            self.obj_seq[i_seq] = [obj_filtered[i].detach() for i in range(self.n_o)]                                    
            self.tiptilt_lr_seq[i_seq] = tiptilt_infer.detach()
            self.tiptilt_hr_seq[i_seq] = tiptilt.detach()
            self.modes_lr_seq[i_seq] = modes_infer.detach()
            self.modes_hr_seq[i_seq] = modes.detach()
                                    
        self.loss = torch.cat(self.loss, dim=0)
        self.tiptilt_lr = torch.cat(self.tiptilt_lr_seq, dim=0)
        self.tiptilt_hr = torch.cat(self.tiptilt_hr_seq, dim=0)
        self.modes_lr = torch.cat(self.modes_lr_seq, dim=0)
        self.modes_hr = torch.cat(self.modes_hr_seq, dim=0)
        
        self.obj = [None] * self.n_o

        for i in range(self.n_o):
            tmp = [self.obj_seq[j][i] for j in range(n_sequences)]
            self.obj[i] = torch.cat(tmp, dim=0)        
            
        return
    
if __name__ == '__main__':
    pass