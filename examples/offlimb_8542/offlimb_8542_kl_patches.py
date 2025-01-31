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

    xy0 = [200, 200]
    lam = 12
    npix = 512
    obs_file = f"../obs/offlimb_20200728_083718_8542_npix512_original.h5"

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
    
    frames = im[:, :, :, 0:npix, 0:npix]

    frames /= np.mean(frames, axis=(-1, -2), keepdims=True)

    frames = torch.tensor(frames.astype('float32'))

    patchify = torchmfbd.Patchify()

    # Patchify the frames
    frames_patches = patchify.patchify(frames, patch_size=64, stride_size=50, flatten_sequences=True)
    
    # frames_back = patchify.unpatchify(frames_patches)
    
    n_scans, n_obj, n_frames, nx, ny = frames.shape

    sigma = torchmfbd.compute_noise(frames_patches)
    sigma = torch.tensor(sigma.astype('float32'))
        
    decSI = torchmfbd.Deconvolution('config_8542_kl.yaml')     

    decSI.deconvolve(frames_patches, 
                     sigma, 
                     infer_object=False, 
                     optimizer='first', 
                     annealing='sigmoid', 
                     simultaneous_sequences=16,
                     n_iterations=10)
        
    modes = decSI.modes.cpu().numpy()
    # psf = patchify.unpatchify(decSI.psf).cpu().numpy()
    wavefront = decSI.wavefront.cpu().numpy()
    degraded = patchify.unpatchify(decSI.degraded, apodization=6).cpu().numpy()
    obj = patchify.unpatchify(decSI.obj, apodization=6).cpu().numpy()
    obj_diffraction = patchify.unpatchify(decSI.obj_diffraction, apodization=6).cpu().numpy()    
    frames = patchify.unpatchify(frames_patches).cpu().numpy()

    fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].imshow(frames[0, 0, 0, :, :])
    ax[1].imshow(obj[0, 0, :, :])    