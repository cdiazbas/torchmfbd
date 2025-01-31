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
            'basis_file': '../basis_nmf/basis_3934_nmf_150_r0_15_50.npz',
            'npix_apod': 12,
            'gpu': 0,
            'npix': 256,
            'ngrid_modes': 16,
            'fourier_filter': True,
            'cutoff': 0.5,
            'n_epochs': 50,
            'lr_obj': 0.2,
            'lr_modes': 0.01,
            'lr_tiptilt': 0.01,
            'lambda_iuwt': [0.0008, 0.00005],
            'lambda_modes': 100.1,
            'lambda_tt': 0.1,
            'iuwt_nbands': 6,
            'weight_obj': [1.0, 0.01],
            'use_checkpointing': False,
            'batch_size': 4,
            'infer_tiptilt': True,
            'infer_modes': True,
            'initial_object': 'contrast'
        }
    
    pl.close('all')

    lam = 14
    xy0 = [200, 200]
    npix = config['npix']
    obs_file = f"obs/spot_20200727_083509_3934_npix512_original.h5"

    if (os.path.exists(obs_file)):
        print(f'Reading observations from {obs_file}...')
        f = h5py.File(obs_file, 'r')
        im = f['im'][:]
        im_d = None
        f.close()
    else:
        root = '/net/diablos/scratch/sesteban/reduc/reduc_andres/spot_20200727_083509_3934'
        label = '20200727_083509_3934_nwav_al'
        print(f'Reading wavelength point {lam}...')
        wb, nb, db = readsst(root, 
                         label, 
                         cam=0, 
                         lam=lam,
                         mod=0, 
                         seq=[0, 1], 
                         xrange=[xy0[0], xy0[0]+npix], 
                         yrange=[xy0[1], xy0[1]+npix], 
                         destretch=False,
                         instrument='CHROMIS')

        ns, nf, nx, ny = wb.shape

        # ns, no, nf, nx, ny
        im = np.concatenate((wb[:, None, ...], nb[:, None, ...]), axis=1)
        im_d = None

        print(f"Saving observations to {obs_file}...")
        f = h5py.File(obs_file, 'w')
        f.create_dataset('im', data=im)
        f.close()

    im = im[:,:, :, 0:config['npix'], 0:config['npix']]
    obj, tiptilt_lr, tiptilt_hr, modes_lr, modes_hr, modes_diffraction, window, loss, psf_modes = SVMOMFBD(im, im_d, config, simultaneous_sequences=1)
    
    root_mfbd = 'aux'
        
    f2 = [fits.open(f'../../spatially_variant/deconvolution/aux/camXXVIII_2020-07-27T08:35:09_00000_12.00ms_G10.00_3934_3934_+65.fits'), fits.open(f'../../spatially_variant/deconvolution/aux/camXXX_2020-07-27T08:35:09_00000_12.00ms_G10.00_3934_3934_+65.fits')]
    
    n = config['npix_apod']


    fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(15, 8))
    
    # WB
    # Frame
    tmp = im[0, 0, 0, 0:config['npix'], 0:config['npix']][n:-n, n:-n]
    contrast = 100 * np.std(tmp) / np.mean(tmp)
    ima = ax[0, 0].imshow(tmp)
    fig.colorbar(ima, ax=ax[0, 0])
    ax[0, 0].set_title(f'Frame {contrast:.2f}%')
    
    # MOMFBD
    tmp = f2[0][0].data[:, :][200-40:200-40+config['npix'], 200-28:200-28+config['npix']][n:-n, n:-n]
    contrast = 100 * np.std(tmp) / np.mean(tmp)
    ima = ax[0, 1].imshow(tmp)
    fig.colorbar(ima, ax=ax[0, 1])
    ax[0, 1].set_title(f'MOMFBD {contrast:.2f}%')

    # SV-MOMFBD
    tmp = obj[0, 0, n:-n, n:-n]
    contrast = 100 * np.std(tmp) / np.mean(tmp)
    ima = ax[0, 2].imshow(tmp)
    fig.colorbar(ima, ax=ax[0, 2])
    ax[0, 2].set_title(f'PCA-MOMFBD {contrast:.2f}%')

    # NB
    # Frame
    tmp = im[0, 1, 0, 0:config['npix'], 0:config['npix']][n:-n, n:-n]
    contrast = 100 * np.std(tmp) / np.mean(tmp)
    ima = ax[1, 0].imshow(tmp)
    fig.colorbar(ima, ax=ax[1, 0])
    ax[1, 0].set_title(f'Frame {contrast:.2f}%')

    # MOMFBD
    tmp = f2[1][0].data[:, :][200-40:200-40+config['npix'], 200-28:200-28+config['npix']][n:-n, n:-n]
    contrast = 100 * np.std(tmp) / np.mean(tmp)
    ima = ax[1, 1].imshow(tmp)
    fig.colorbar(ima, ax=ax[1, 1])
    ax[1, 1].set_title(f'MOMFBD: {contrast:.2f}%')

    # SV-MOMFBD
    tmp = obj[0, 1, n:-n, n:-n]
    contrast = 100 * np.std(tmp) / np.mean(tmp)
    ima = ax[1, 2].imshow(tmp)
    fig.colorbar(ima, ax=ax[1, 2])
    ax[1, 2].set_title(f'PCA-MOMFBD {contrast:.2f}%')

    k_im, power_im = az_average.power_spectrum(im[0, 0, 0, 0:config['npix'], 0:config['npix']][n:-n, n:-n])
    k, power = az_average.power_spectrum(obj[0, 0, n:-n, n:-n])
    k_old, power_old = az_average.power_spectrum(f2[0][0].data[:, ::-1][200-20:200-20+config['npix'], 200-14:200-14+config['npix']][n:-n, n:-n])

    fig, ax = pl.subplots()
    ax.semilogy(k_im, power_im / power_im[0], label=r'1$^{st}$ image', linewidth=4.0)
    ax.semilogy(k, power / power[0], label='SV-MOMFBD', linewidth=4.0)
    ax.semilogy(k_old, power_old / power_old[0], label='MOMFBD', linewidth=4.0)
    ax.set_xlabel(r'Wavenumber [px$^{-1}$]', fontsize=16)
    ax.set_ylabel(r'Azimuthally averaged power spectrum', fontsize=16)
    ax.legend()
    ax.set_title('Sunspot')

    n_grid = config['ngrid_modes']
    psf_pix = 20
    psf_modes_sh = np.fft.fftshift(psf_modes, axes=(1, 2))[:, 128-psf_pix:128+psf_pix, 128-psf_pix:128+psf_pix]
    out = np.sum(modes_lr[0, 0, :, :, :][:, :, :, None, None] * psf_modes_sh[:, None, None, :, :], axis=0)
    out = out.transpose(0, 2, 1, 3).reshape((n_grid * 2 * psf_pix, n_grid * 2 * psf_pix))
