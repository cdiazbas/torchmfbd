import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
import torchmfbd
import torch
import platform

if __name__ == '__main__':
    config = {            
            'telescope': {
                'diameter': 400.0,                
                'central_obscuration': 0.0,                
            },
            'images': {                
                'n_pixel' : 801,
                'wavelength': 5500.0,            
                'pix_size': 0.0128,
                'apodization_border': 6,
            },

            'psf': {
                'model': 'wavefront',
            },
            
            'optimization': {
                'gpu': 0,                
                'transform': 'softplus',
                'loss': 'momfbd',
                'lr_obj': 0.02,
                'lr_modes': 0.08,
            },

            'regularization': {                
                'lambda_iuwt': 0.0,
                'iuwt_nbands': 5,
            },            
            
            'filter': {
                'image_filter': 'gaussian',
                'cutoff': [0.85],
            },            
            
            'modes': {
                'nmax_modes': 25,
                'basis': 'kl',
            },            
            
            'initialization': {
                'object': 'contrast',
                'modes_std': 0.0,
            },

            'annealing': {
                'start_pct': 0.0,
                'end_pct': 0.85,
            }
        }
    
    pl.close('all')

    if platform.node() == 'linux':
        root = '../../collab/luzma'
    if platform.node() == 'gpu1':
        root = '../est_momfbd'

    full_res = fits.open(f'{root}/images/crop_Im_corr.fits')[0].data
    full_res /= np.mean(full_res)

    f = fits.open(f"{root}/images/scicam_20241108_154203_cube_100.fits")
    dark = fits.open(f"{root}/images/dark_20241108_155534_0crop.fits")[0].data
    flat = fits.open(f"{root}/images/flat_20241108_155514_0_CROP.fits")[0].data

    nx, ny = dark.shape

    nframes = 50

    frames = np.zeros((1, 1, nframes, nx, ny))

    for i in range(nframes):
        frames[0, 0, i, :, :] = (f[i].data[0, :, :] - dark) / flat    

    frames /= np.mean(frames, axis=(-1, -2), keepdims=True)

    n_scans, n_obj, n_frames, nx, ny = frames.shape           

    frames = torch.tensor(frames.astype('float32'))
    
    
    config['images']['n_pixel'] = frames.shape[-1]

    sigma = torchmfbd.compute_noise(frames)
    sigma = torch.tensor(sigma.astype('float32'))
        
    deconvolution = torchmfbd.Deconvolution(config)      
    
    deconvolution.deconvolve(frames, 
                             sigma, 
                             infer_object=False, 
                             optimizer='first', 
                             annealing='sigmoid', 
                             simultaneous_sequences=1,
                             n_iterations=100)
    
    frame0_apodized = deconvolution.apodize(frames[0, 0, 0, :, :]).cpu().numpy()
    full_res_th = torch.tensor(full_res.astype('float32'))
    full_res_apodized = deconvolution.apodize(full_res_th).cpu().numpy()    

    modes = deconvolution.modes.cpu().numpy()
    psf = deconvolution.psf.cpu().numpy()
    wavefront = deconvolution.wavefront.cpu().numpy()
    degraded = deconvolution.degraded.cpu().numpy()
    reconstructed = deconvolution.reconstructed.cpu().numpy()
    reconstructed_diffraction = deconvolution.reconstructed_diffraction.cpu().numpy()
    loss = deconvolution.loss.cpu().numpy()
    contrast = deconvolution.contrast.cpu().numpy()
    
    
    # # modes = torch.tensor(modes)
    # # reconstructed = torch.tensor(reconstructed)
    
    # # modes, psf, wavefront, degraded, reconstructed, reconstructed_diff, loss, contrast = classic.deconvolve(frames, 
    # #                                                                                                         sigma, 
    # #                                                                                                         infer_object=True,
    # #                                                                                                         optimizer='second',
    # #                                                                                                         obj_in=reconstructed,                                                                                                            
    # #                                                                                                         modes_in=modes,
    # #                                                                                                         annealing=False,
    # #                                                                                                         n_iterations=10)
    
    frames = frames.cpu().numpy()

    vmin = np.min(full_res)
    vmax = np.max(full_res)

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(15, 15))
    
    var = [frames[0, 0, 0, :, :], frames[0, 0, 1, :, :], full_res, reconstructed[0, 0, :, :]]
    labels = ['Frame 1', 'Frame 2', 'Sun', 'Reconstructed']

    for i in range(4):
        contrast = np.std(var[i]) / np.mean(var[i]) * 100.0
        im = ax.flat[i].imshow(var[i], vmin=vmin, vmax=vmax)
        pl.colorbar(im, ax=ax.flat[i])
        ax.flat[i].set_title(f'{labels[i]} - {contrast:5.2f}%')

    f_rec, power_rec = deconvolution.azimuthal_power(reconstructed[0, 0, :, :])
    f_frame, power_frame = deconvolution.azimuthal_power(frame0_apodized)
    f_sun, power_sun = deconvolution.azimuthal_power(full_res_apodized)

    fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax[0].loglog(f_rec, power_rec / power_rec[1], label='Reconstructed')
    ax[0].loglog(f_frame, power_frame / power_frame[1], label='Frame 1')
    ax[0].loglog(f_sun, power_sun / power_sun[1], label='Sun')
    ax[0].legend()
    ax[0].set_ylim([1e-7, 10.0])

    ax[1].loglog(1.0 / f_rec * 0.0128, power_rec / power_rec[1], label='Reconstructed')
    ax[1].loglog(1.0 / f_frame * 0.0128, power_frame / power_frame[1], label='Frame 1')
    ax[1].loglog(1.0 / f_sun * 0.0128, power_sun / power_sun[1], label='Sun')
    ax[1].legend()
    ax[1].set_ylim([1e-7, 10.0])

    # # npix = 80
    # # fig, ax = pl.subplots(nrows=2, ncols=12, figsize=(25, 6))
    # # for i in range(12):
    # #     psf_inferred = np.fft.fftshift(psf[0, i, :, :])
    # #     psf_real = np.fft.fftshift(f[0].data[i, :, :])
    # #     ax[0, i].imshow(np.sqrt(psf_inferred[576-npix: 576+npix, 576-npix: 576+npix]))
    # #     ax[1, i].imshow(np.sqrt(psf_real[576-npix: 576+npix, 576-npix: 576+npix]))
    # # ax[0, 0].set_title('Inferred PSF')
    # # ax[1, 0].set_title('Real PSF')
