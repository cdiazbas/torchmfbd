import numpy as np
import os
import h5py
import torch
import matplotlib.pyplot as pl
import sys
sys.path.append('../')
from readsst import readsst
import torchmfbd
import psf_tools


if __name__ == '__main__':

    xy0 = [200, 200]
    lam = 7
    npixx = 256
    npixy = 512
    # the burst as downloaded, or another one with the same layout (e.g. the output of ../spot_8542_destrech)
    obs_file = sys.argv[1] if len(sys.argv) > 1 else "spot_20200727_083509_8542_npix512_original.h5"

    print(f'Reading observations from {obs_file}...')
    f = h5py.File(obs_file, 'r')
    im = f['im'][:]    
    f.close()
    
    frames = im[:, :, :, 0:npixx, 0:npixy]

    frames /= np.mean(frames, axis=(-1, -2), keepdims=True)

    frames = torch.tensor(frames.astype('float32'))

    patchify = torchmfbd.Patchify4D()    
            
    n_scans, n_obj, n_frames, nx, ny = frames.shape
    
    decSI = torchmfbd.Deconvolution('kl.yaml')
    
    # Patchify and add the frames
    for i in range(2):        
        frames_patches = patchify.patchify(frames[:, i, :, :, :], patch_size=128, stride_size=32, flatten_sequences=True)   # n_pixel of kl.yaml
        decSI.add_frames(frames_patches, id_object=i, id_diversity=0, diversity=0.0)
    decSI.set_neighbours(patchify.neighbours())          # for the coupling regularization of kl.yaml
                
    decSI.deconvolve(infer_object=False, 
                     optimizer='lbfgs', 
                     simultaneous_sequences=frames_patches.shape[0],     # all patches (65 here) in one batch, for the coupling of kl.yaml
                     n_iterations=10)
            
    obj = []
    for i in range(2):
        obj.append(patchify.unpatchify(decSI.obj[i], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())        

    nx, ny = obj[0].shape[1], obj[0].shape[2]
    
    fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(15, 10))
    for i in range(2):
        ax[i, 0].imshow(frames[0, i, 0, 0:, 0:], cmap='gray', interpolation='nearest')
        ax[i, 1].imshow(obj[i][0, 0:, 0:], cmap='gray', interpolation='nearest')


    decSI.update_object(cutoffs=[[0.3, 0.35], [0.3, 0.35]])

    # Unpatchify
    obj = []
    for i in range(2):
        obj.append(patchify.unpatchify(decSI.obj[i], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())        
        
    for i in range(2):
        ax[i, 2].imshow(obj[i][0, 0:, 0:], cmap='gray', interpolation='nearest')

    # Force same vmin and vmax for all images using the original images as reference:
    vmin = np.min(frames[:, :, 0, :, :].cpu().numpy())
    vmax = np.max(frames[:, :, 0, :, :].cpu().numpy())
    for i in range(2):
        for j in range(3):
            ax[i, j].get_images()[0].set_clim(vmin, vmax)

    ax[0, 1].set_title('Reconstructed object')
    ax[0, 2].set_title('Reconstructed object (updated cutoffs)')
    pl.savefig('spot_8542_kl_patches.png', dpi=300, bbox_inches='tight')

    # Export and plot PSFs for comparison
    psfs = psf_tools.export_psfs(decSI)
    psf_tools.plot_psfs(psfs, model_name='KL')


