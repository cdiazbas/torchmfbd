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
    
    keep_time = True

    # The series is fitted in windows of consecutive steps: the time term ties the steps of a window, and all the patches
    # of a window must fit in one batch on the GPU for the coupling (49 steps at once need more than 24 GB). Longer
    # windows did not help on the real-time benchmark (8 steps at once: 2.7 % worse than two windows of 4; 7 not measured)
    # The windows overlap, and each step is the weighted mean of the windows that contain it, with a weight that
    # falls towards the window edges: steps fitted at the edge of a window have one neighbour, and taking each step from
    # a single window left jumps across window borders (rms change between consecutive objects 0.21 across borders
    # against 0.09 inside without overlap; 0.14 when each step comes from its nearest window)
    window = 7 if keep_time else n_scans
    advance = 4 if keep_time else n_scans
    starts = list(range(0, max(n_scans - window, 0) + 1, advance))
    if starts[-1] + window < n_scans:
        starts.append(n_scans - window)
    acc_obj, acc_bf, acc_w = [None, None], [None, None], np.zeros(n_scans)
    obj = [[], []]
    best_frame = [[], []]
    for w, t0 in enumerate(starts):
        fr = frames[t0:t0 + window]
        n_t = fr.shape[0] if keep_time else 1
        k = np.arange(fr.shape[0])
        wt = np.minimum(k + 1, fr.shape[0] - k).astype(float)        # 1 at the edges, largest in the centre
        if t0 == 0:
            wt[:fr.shape[0] // 2] = wt.max()                          # the first and last steps of the series have no
        if t0 + fr.shape[0] == n_scans:                                # other window: full weight there
            wt[fr.shape[0] // 2:] = wt.max()

        if regularize:
            decSI = torchmfbd.Deconvolution('spot_8542_joint_time.yaml')
        else:
            decSI = torchmfbd.Deconvolution('spot_8542_joint.yaml')

        # Patchify and add the frames
        frames_patches = [None] * 2
        for i in range(2):
            frames_patches[i] = patchify.patchify(fr[:, i, :, :, :], patch_size=128, stride_size=32, flatten_sequences=not keep_time)   # n_pixel of the yaml
            if keep_time:
                decSI.add_frames(frames_patches[i].transpose(0, 1), id_object=i, id_diversity=0, diversity=0.0, sigma=0.015)
            else:
                decSI.add_frames(frames_patches[i], id_object=i, id_diversity=0, diversity=0.0, sigma=0.015)

        # For the coupling regularization of the yaml: with time, the sequences of the 5-D data are the patches of one
        # image, so the pairs are those of a single step (the coupling acts within each step)
        one_step = torchmfbd.Patchify4D()
        one_step.patchify(fr[0:1, 0, :, :, :], patch_size=128, stride_size=32, flatten_sequences=True)
        decSI.set_neighbours(one_step.neighbours())
        n_patches = one_step.n_patches

        decSI.deconvolve(infer_object=False,
                         optimizer='lbfgs',
                         simultaneous_sequences=n_patches * n_t if keep_time else n_patches * fr.shape[0],   # whole window in one batch
                         n_iterations=10)

        for i in range(2):
            if keep_time:
                obj[i].append(patchify.unpatchify(decSI.obj[i].transpose(0, 1).unsqueeze(2), apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
                best_frame[i].append(patchify.unpatchify(frames_patches[i][:, :, ind_best_contrast:ind_best_contrast+1, :, :], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
            else:
                obj[i].append(patchify.unpatchify(decSI.obj[i][:,None,...], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
                best_frame[i].append(patchify.unpatchify(frames_patches[i][:, ind_best_contrast:ind_best_contrast+1, :, :], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
            o, b_ = obj[i].pop(), best_frame[i].pop()
            if acc_obj[i] is None:
                acc_obj[i] = np.zeros((n_scans,) + o.shape[1:]); acc_bf[i] = np.zeros((n_scans,) + b_.shape[1:])
            acc_obj[i][t0:t0 + fr.shape[0]] += wt[:, None, None, None] * o
            acc_bf[i][t0:t0 + fr.shape[0]] += wt[:, None, None, None] * b_
        acc_w[t0:t0 + fr.shape[0]] += wt
        print(f'window {t0}-{t0 + fr.shape[0] - 1} done')

    obj = [a / acc_w[:, None, None, None] for a in acc_obj]
    best_frame = [a / acc_w[:, None, None, None] for a in acc_bf]

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