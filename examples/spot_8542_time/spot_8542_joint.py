import numpy as np
import os
import h5py
import torch
import matplotlib.pyplot as pl
import sys
sys.path.append('../')
from readsst import readsst
import torchmfbd
from astropy.io import fits


if __name__ == '__main__':


    regularize = True

    pl.close('all')

    npix = 256

    obs_file = 'sequence_8542_npix256_aligned.h5'
    print(f'Reading observations from {obs_file}...')
    f = h5py.File(obs_file, 'r')
    im = f['im'][:]
    im_d = None
    f.close()
        
    frames = im[0:49, :, :, 0:npix, 0:npix]
    
    frames /= np.mean(frames, axis=(-1, -2), keepdims=True)

    contrast = np.std(frames, axis=(-1,-2)) / np.mean(frames, axis=(-1,-2))
    ind_best_contrast = np.argmax(contrast[0, 0, :])

    frames = torch.tensor(frames.astype('float32'))

    patchify = torchmfbd.Patchify4D()    
            
    n_scans, n_obj, n_frames, nx, ny = frames.shape
    
    if regularize:
        decSI = torchmfbd.Deconvolution('spot_8542_joint_time.yaml')
    else:
        decSI = torchmfbd.Deconvolution('spot_8542_joint.yaml')

    keep_time = True

    if keep_time:
        flatten_sequences = False
        n_t = frames.shape[0]
    else:
        flatten_sequences = True
        n_t = 1

    # Patchify and add the frames
    frames_patches = [None] * 2
    for i in range(2):        
        frames_patches[i] = patchify.patchify(frames[:, i, :, :, :], patch_size=92, stride_size=40, flatten_sequences=flatten_sequences)                
        if keep_time:
            decSI.add_frames(frames_patches[i].transpose(0, 1), id_object=i, id_diversity=0, diversity=0.0, sigma=0.015)
        else:
            decSI.add_frames(frames_patches[i], id_object=i, id_diversity=0, diversity=0.0, sigma=0.015)
        
    decSI.deconvolve(infer_object=False, 
                     optimizer='adam',                      
                     simultaneous_sequences=49*5,
                     n_iterations=50)
        
    best_frame = []
    obj = []
    for i in range(2):
        if keep_time:
            obj.append(patchify.unpatchify(decSI.obj[i].transpose(0, 1).unsqueeze(2), apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
            best_frame.append(patchify.unpatchify(frames_patches[i][:, :, ind_best_contrast:ind_best_contrast+1, :, :], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
        else:
            obj.append(patchify.unpatchify(decSI.obj[i][:,None,...], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())        
            best_frame.append(patchify.unpatchify(frames_patches[i][:, ind_best_contrast:ind_best_contrast+1, :, :], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
    

    # fig, ax = pl.subplots(ncols=4, nrows=n_t, figsize=(16, 4*n_t), sharex=True, sharey=True)

    if regularize:
        # Named after the time regularization of the configuration, as movie.py reads it
        lambda_t = decSI.config['regularization']['time']['lambda']
        contrast = 'contrast_' if decSI.config['optimization']['filter_loss'] == 'wiener_time_contrast' else ''
        fout = h5py.File(f'reconstruction_regularized_{contrast}{lambda_t:g}.h5', 'w')
    else:
        fout = h5py.File('reconstruction_original.h5', 'w')
    for io in range(2):
        # for j in range(n_t):
            # ax[j, 2*io].imshow(best_frame[io][j, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
            # ax[j, 2*io+1].imshow(obj[io][j, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)    
            # ax[j, 2*io].set_title(f't={j}, obj={io} best frame')
            # ax[j, 2*io+1].set_title(f't={j}, obj={io} deconvolved')
        
        fout.create_dataset(f'obj_{io}', data=obj[io])
        fout.create_dataset(f'best_frame_{io}', data=best_frame[io])
    fout.close()

    pl.tight_layout()    