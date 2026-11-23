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

    # Optional second pass that follows the solar motion between steps (see below); off by default
    two_pass_flow = False

    # The series is fitted in windows of consecutive steps: the time term ties the steps of a window, and all the patches
    # of a window must fit in one batch on the GPU for the coupling (49 steps at once need more than 24 GB). Each step is
    # taken from the window of 5 steps centred nearest to it, the windows placed every 3 steps on the absolute step index
    # (torchmfbd.plan_time_windows), so a step comes out the same however the series is cut; a step whose frames are below
    # the noise gets a window that reaches the nearest step with SNR >= 12. With the SNR rule of the time weight, which
    # the library applies by default, against the windows of 7 every 4 with weighted means used here until 2026-11 (git
    # tag time-example-7of4): better in 12 of 12 synthetic 49-step series with a known truth (mean object error -23.7 %),
    # and the very bad step 16 of this sequence restored (wide-band contrast 0.065 -> 0.087, the other steps unchanged)
    last = {}                                             # the configuration of the fits, for the output file name
    if keep_time:
        # SNR of each step from its wide-band frames: diffraction cutoff of the 1-m telescope at 8542 A, 0.059"/px
        cutoff = 0.059 / (8542e-10 / 1.0 * 206265.0)       # cycles per pixel
        f = np.fft.fftfreq(nx)
        rho = torch.tensor(np.hypot(*np.meshgrid(f, f, indexing='ij')) / cutoff)
        snr = torchmfbd.step_snr(frames[:, 0].double(), rho).numpy()
        plan = torchmfbd.plan_time_windows(snr)
    else:
        plan = [(0, n_scans)] * n_scans                   # no time term: every step in one fit, as before

    def fit(fr):
        """Fit the frames of one window; the objects and best frames of its steps, (T, 1, n, n) per object."""
        n_t = fr.shape[0] if keep_time else 1
        if regularize:
            decSI = torchmfbd.Deconvolution('spot_8542_joint_time.yaml')
        else:
            decSI = torchmfbd.Deconvolution('spot_8542_joint.yaml')

        # Patchify and add the frames
        frames_patches = [None] * 2
        for i in range(2):
            frames_patches[i] = patchify.patchify(fr[:, i, :, :, :], patch_size=128, stride_size=64, flatten_sequences=not keep_time)   # n_pixel of the yaml; stride 64: see the yaml
            if keep_time:
                decSI.add_frames(frames_patches[i].transpose(0, 1), id_object=i, id_diversity=0, diversity=0.0, sigma=0.015)
            else:
                decSI.add_frames(frames_patches[i], id_object=i, id_diversity=0, diversity=0.0, sigma=0.015)

        # For the coupling regularization of the yaml: with time, the sequences of the 5-D data are the patches of
        # one image, so the pairs are those of a single step (the coupling acts within each step)
        one_step = torchmfbd.Patchify4D()
        one_step.patchify(fr[0:1, 0, :, :, :], patch_size=128, stride_size=64, flatten_sequences=True)
        decSI.set_neighbours(one_step.neighbours())
        n_patches = one_step.n_patches
        last['config'] = decSI.config

        decSI.deconvolve(infer_object=False,
                         optimizer='lbfgs',
                         simultaneous_sequences=n_patches * n_t if keep_time else n_patches * fr.shape[0],   # whole window in one batch
                         n_iterations=10)

        objs, bfs = [], []
        for i in range(2):
            if keep_time:
                objs.append(patchify.unpatchify(decSI.obj[i].transpose(0, 1).unsqueeze(2), apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
                bfs.append(patchify.unpatchify(frames_patches[i][:, :, ind_best_contrast:ind_best_contrast+1, :, :], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
            else:
                objs.append(patchify.unpatchify(decSI.obj[i][:,None,...], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
                bfs.append(patchify.unpatchify(frames_patches[i][:, ind_best_contrast:ind_best_contrast+1, :, :], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
        return objs, bfs


    def fit_window(t0, n):
        """The objects and best frames of steps t0 ... t0 + n - 1, fitted together."""
        fr = frames[t0:t0 + n]
        objs, bfs = fit(fr)

        if two_pass_flow and keep_time:
            # Second pass on frames aligned to the average geometry of the window: the flow of the wide-band objects of
            # the first pass (the same for both objects), the frames warped with it, fitted again, and the objects
            # warped back. On the real-time benchmark: -2.8 / -3.8 % object error on hold-out scenes (6 of 6), +0.4 %
            # on the default ones of one GPU; twice the time
            ob = torch.tensor(objs[0][None, :, 0].astype('float32'))                       # (1, T, m, m)
            _, tt = torchmfbd.destretch(ob[:, None], ngrid=16, lr=0.05, reference_frame=0, border=8, n_iterations=60,
                                        lambda_tt=0.01, average_geometry=True, n_refine=2)   # (1, T, 2, m, m)
            m = ob.shape[-1]; off = (nx - m) // 2
            tt_full = torch.nn.functional.pad(tt[0], (off, ny - m - off, off, nx - m - off), mode='replicate')
            warped = torch.stack([torchmfbd.apply_destretch(fr[:, i].reshape(1, 1, -1, nx, ny), tt_full.repeat_interleave(n_frames, 0))[0, 0].reshape(-1, n_frames, nx, ny)
                                  for i in range(2)], dim=1)
            objs2, _ = fit(warped)
            objs = [torchmfbd.apply_destretch(torch.tensor(o[None, :, 0][:, None].astype('float32')), -tt)[0, 0][:, None].numpy() for o in objs2]

        print(f'window {t0}-{t0 + n - 1} done')
        return [([objs[i][t] for i in range(2)], [bfs[i][t] for i in range(2)]) for t in range(n)]

    steps = torchmfbd.assemble_time_windows(plan, fit_window)
    obj = [np.stack([s[0][i] for s in steps]) for i in range(2)]
    best_frame = [np.stack([s[1][i] for s in steps]) for i in range(2)]

    # fig, ax = pl.subplots(ncols=4, nrows=n_t, figsize=(16, 4*n_t), sharex=True, sharey=True)

    if regularize:
        # Named after the time regularization of the configuration, as movie.py reads it
        lambda_t = last['config']['regularization']['time']['lambda']
        contrast = 'contrast_' if last['config']['optimization']['filter_loss'] == 'wiener_time_contrast' else ''
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