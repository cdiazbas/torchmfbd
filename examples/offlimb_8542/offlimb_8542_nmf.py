from svmomfbd import SVMOMFBD
import numpy as np
import os
import h5py
from einops import rearrange
import matplotlib.pyplot as pl
import az_average
from astropy.io import fits
from readsst import readsst


if __name__ == '__main__':

    config = {
            'n_modes': 150,
            'basis_type': 'nmf',
            'basis_file': '../basis_nmf/basis_torch_8542_nmf_150_r0_5_30.npz',
            'npix_apod': 12,
            'gpu': 0,
            'npix': 256,
            'ngrid_modes': 8,
            'fourier_filter': True,
            'cutoff': 0.75,
            'n_epochs': 50,
            'lr_obj': 0.02,
            'lr_modes': 0.003,
            'lr_tiptilt': 0.003,            
            'lambda_iuwt': [0.002, 0.0001],
            'lambda_modes': 10.5,
            'lambda_tt': 0.1,
            'iuwt_nbands': 6,
            'weight_obj': [0.01, 1.0],
            'use_checkpointing': False,
            'batch_size': 4,
            'infer_tiptilt': True,
            'infer_modes': True,
            'initial_object': 'contrast'
        }
            
    pl.close('all')

    xy0 = [200, 200]
    lam = 12
    npix = config['npix']
    obs_file = f"obs/offlimb_20200728_083718_8542_npix512_original.h5"

    if (os.path.exists(obs_file)):
        print(f'Reading observations from {obs_file}...')
        f = h5py.File(obs_file, 'r')
        im = f['im'][:]
        im_d = None
        f.close()
    else:
        root = '/net/diablos/scratch/sesteban/reduc/reduc_andres/limb_20200728_083718'
        label = '20200728_083718_8542_nwav_al'
        print(f'Reading wavelength point {lam}...')
        wb, nb = readsst(root, 
                         label, 
                         cam=0, 
                         lam=lam, 
                         mod=0, 
                         seq=[9, 10], 
                         xrange=[xy0[0], xy0[0]+npix], 
                         yrange=[xy0[1], xy0[1]+npix], 
                         destretch=False,
                         instrument='CRISP')

        ns, nf, nx, ny = wb.shape

        # ns, no, nf, nx, ny
        im = np.concatenate((wb[:, None, ...], nb[:, None, ...]), axis=1)
        im_d = None

        print(f"Saving observations to {obs_file}...")
        f = h5py.File(obs_file, 'w')
        f.create_dataset('im', data=im)
        f.close()
    
    im = im[:, :, :, 0:config['npix'], 0:config['npix']]
    obj, tiptilt_lr, tiptilt_hr, modes_lr, modes_hr, modes_diffraction, window, loss, psf_modes = SVMOMFBD(im, im_d, config, simultaneous_sequences=1)
    
    root_mfbd = 'aux'
        
    f2 = [fits.open(f'../../spatially_variant/deconvolution/aux/camXX_2020-07-27T08:35:09_00000_8542_8542_+65_lc0.fits'), fits.open(f'../../spatially_variant/deconvolution/aux/camXIX_2020-07-27T08:35:09_00000_8542_8542_+65_lc0.fits')]

    n = config['npix_apod']

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 8))
    
    # WB
    # Frame
    tmp = im[0, 0, 0, 0:config['npix'], 0:config['npix']][n:-n, n:-n]
    contrast = 100 * np.std(tmp) / np.mean(tmp)
    ima = ax[0, 0].imshow(tmp)
    fig.colorbar(ima, ax=ax[0, 0])
    ax[0, 0].set_title(f'Frame {contrast:.2f}%')
    

    # SV-MOMFBD
    tmp = obj[0, 0, n:-n, n:-n]
    contrast = 100 * np.std(tmp) / np.mean(tmp)
    ima = ax[0, 1].imshow(tmp)
    fig.colorbar(ima, ax=ax[0, 1])
    ax[0, 1].set_title(f'NMF-MOMFBD {contrast:.2f}%')

    # NB
    # Frame
    tmp = im[0, 1, 0, 0:config['npix'], 0:config['npix']][n:-n, n:-n]
    contrast = 100 * np.std(tmp) / np.mean(tmp)
    ima = ax[1, 0].imshow(tmp)
    fig.colorbar(ima, ax=ax[1, 0])
    ax[1, 0].set_title(f'Frame {contrast:.2f}%')

    # SV-MOMFBD
    tmp = obj[0, 1, n:-n, n:-n]
    contrast = 100 * np.std(tmp) / np.mean(tmp)
    ima = ax[1, 1].imshow(tmp)
    fig.colorbar(ima, ax=ax[1, 1])
    ax[1, 1].set_title(f'NMFs-MOMFBD {contrast:.2f}%')

    k_im, power_im = az_average.power_spectrum(im[0, 1, 0, 0:config['npix'], 0:config['npix']][n:-n, n:-n])
    k, power = az_average.power_spectrum(obj[0, 1, n:-n, n:-n])
    
    fig, ax = pl.subplots()
    ax.semilogy(k_im, power_im / power_im[0], label=r'1$^{st}$ image', linewidth=4.0)
    ax.semilogy(k, power / power[0], label='SV-MOMFBD', linewidth=4.0)
    ax.set_xlabel(r'Wavenumber [px$^{-1}$]', fontsize=16)
    ax.set_ylabel(r'Azimuthally averaged power spectrum', fontsize=16)
    ax.legend()
    ax.set_title('Offlimb')

    n_grid = config['ngrid_modes']
    psf_pix = 20
    psf_modes_sh = np.fft.fftshift(psf_modes, axes=(1, 2))[:, 128-psf_pix:128+psf_pix, 128-psf_pix:128+psf_pix]
    out = np.sum(modes_lr[0, 0, :, :, :][:, :, :, None, None] * psf_modes_sh[:, None, None, :, :], axis=0)
    out = out.transpose(0, 2, 1, 3).reshape((n_grid * 2 * psf_pix, n_grid * 2 * psf_pix))


