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
    lam = 7
    npix = 512
    obs_file = f"spot_20200727_083509_8542_npix512_original.h5"

    print(f'Reading observations from {obs_file}...')
    f = h5py.File(obs_file, 'r')
    im = f['im'][:]
    im_d = None
    f.close()
    
    frames = im[:, :, :, 0:npix, 0:npix]

    frames /= np.mean(frames, axis=(-1, -2), keepdims=True)

    frames = torch.tensor(frames.astype('float32'))

    patchify = torchmfbd.Patchify4D()    
            
    n_scans, n_obj, n_frames, nx, ny = frames.shape
    
    decSI = torchmfbd.Deconvolution('kl.yaml')

    # Patchify and add the frames
    for i in range(2):        
        frames_patches = patchify.patchify(frames[:, i, :, :, :], patch_size=64, stride_size=50, flatten_sequences=True)
        decSI.add_frames(frames_patches, id_object=i, id_diversity=0, diversity=0.0)
            
    
    decSI.deconvolve(infer_object=False, 
                     optimizer='first', 
                     simultaneous_sequences=16,
                     n_iterations=100)
        
    # modes = decSI.modes.cpu().numpy()
    # # psf = patchify.unpatchify(decSI.psf).cpu().numpy()
    # wavefront = decSI.wavefront.cpu().numpy()
    # degraded = patchify.unpatchify(decSI.degraded, apodization=6).cpu().numpy()
    obj = []
    for i in range(2):
        obj.append(patchify.unpatchify(decSI.obj[i], apodization=6).cpu().numpy())        
    # obj_diffraction = patchify.unpatchify(decSI.obj_diffraction, apodization=6).cpu().numpy()    
    # frames = patchify.unpatchify(frames_patches).cpu().numpy()

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i in range(2):
        ax[0, i].imshow(frames[0, i, 0, :, :])
        ax[1, i].imshow(obj[i][0, :, :])