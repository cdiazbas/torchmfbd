import numpy as np
import os
import h5py
import torch
import matplotlib.pyplot as pl
import sys
sys.path.append('../')
from readsst import readsst
import torchmfbd


if __name__ == '__main__':

    lam = 7
    xy0 = [200, 200]
    npix = 512
    obs_file = f"../obs/qs_20190801_081547_8542_npix512_original.h5"

    if (os.path.exists(obs_file)):
        print(f'Reading observations from {obs_file}...')
        f = h5py.File(obs_file, 'r')
        im = f['im'][:]
        im_d = None
        f.close()
    else:
        root = '/net/diablos/scratch/sesteban/reduc/reduc_andres/qs_20190801_081547'
        label = '20190801_081547_nwav_al'
        print(f'Reading wavelength point {lam}...')
        wb, nb = readsst(root, 
                         label, 
                         cam=0, 
                         lam=lam, 
                         mod=0, 
                         seq=[10, 11], 
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
    
    frames = im[:, :, :, 0:npix, 0:npix]

    frames /= np.mean(frames, axis=(-1, -2), keepdims=True)

    frames = torch.tensor(frames.astype('float32'))

    patchify = torchmfbd.Patchify()

    # Patchify the frames
    frames_patches = patchify.patchify(frames, patch_size=64, stride_size=50, flatten_sequences=True)

    n_scans, n_obj, n_frames, nx, ny = frames.shape

    config = {            
            'telescope': {
                'diameter': 100.0,                
                'central_obscuration': 0.0,                
            },
            'images': {                
                'n_pixel' : 64,
                'wavelength': 8542.0,            
                'pix_size': 0.059,
                'apodization_border': 6,
            },
            
            'optimization': {
                'gpu': 0,                
                'transform': 'softplus',
                'softplus_scale': 1.0,
                'loss': 'momfbd',
                'lr_obj': 0.02,
                'lr_modes': 0.08,
            },

            'regularization': {                
                'iuwt1': {
                    'variable': 'object',
                    'lambda': 0.03,
                    'nbands': 5,
                },
            },            
            
            'filter': {
                'image_filter': 'gaussian',
                'cutoff': [0.85, 0.85],
            },            
            
            'psf': {
                'model': 'kl',
                'nmax_modes': 44,            
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

    sigma = torchmfbd.compute_noise(frames_patches)
    sigma = torch.tensor(sigma.astype('float32'))
        
    decSI = torchmfbd.Deconvolution(config)     

    decSI.deconvolve(frames_patches, 
                             sigma, 
                             infer_object=False, 
                             optimizer='first', 
                             annealing='sigmoid', 
                             simultaneous_sequences=1,
                             n_iterations=20)
    
    modes = decSI.modes.cpu().numpy()
    # psf = patchify.unpatchify(decSI.psf).cpu().numpy()
    wavefront = decSI.wavefront.cpu().numpy()
    degraded = patchify.unpatchify(decSI.degraded, apodization=6).cpu().numpy()
    obj = patchify.unpatchify(decSI.obj, apodization=6).cpu().numpy()
    obj_diffraction = patchify.unpatchify(decSI.obj_diffraction, apodization=6).cpu().numpy()    
    frames = patchify.unpatchify(frames_patches).cpu().numpy()

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(15, 15))
    ax[0, 0].imshow(frames[0, 0, 0, :, :])
    ax[0, 1].imshow(obj[0, 0, :, :])
    ax[1, 0].imshow(degraded[0, 0, 0, :, :])
    ax[1, 1].imshow(obj_diffraction[0, 0, :, :])