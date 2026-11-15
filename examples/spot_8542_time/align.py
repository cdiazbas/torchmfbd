import numpy as np
import os
import h5py
import torch
import matplotlib.pyplot as pl
from matplotlib import animation
import sys
sys.path.append('../')
from readsst import readsst
import torchmfbd
from astropy.io import fits
from einops import rearrange

if __name__ == '__main__':

    lam = 6
    npix = 256
    xy0 = [150, 150]

    obs_file = 'sequence_8542_npix256_aligned.h5'
    
    # Folder of the SST reduction: first argument, or the SST_REDUC_ROOT environment variable
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('SST_REDUC_ROOT')
    if root is None:
        sys.exit('usage: python align.py <reduction folder of spot_20200727_083509_8542> (or set SST_REDUC_ROOT)')
    label = '20200727_083509_8542_nwav_al'
    print(f'Reading wavelength point {lam}...')
    for i in range(49):
        wb_tmp, nb_tmp = readsst(root, 
                            label, 
                            cam=0, 
                            lam=lam, 
                            mod=0, 
                            seq=[i, i+1],
                            xrange=[xy0[0], xy0[0]+npix], 
                            yrange=[xy0[1], xy0[1]+npix], 
                            destretch=False,
                            instrument='CRISP')
        
        if i == 0:
            wb = wb_tmp
            nb = nb_tmp
        if i > 0:
            wb = np.concatenate((wb, wb_tmp), axis=0)
            nb = np.concatenate((nb, nb_tmp), axis=0)

    ns, nf, nx, ny = wb.shape

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    wb = torch.tensor(wb, dtype=torch.float32).to(device)
    nb = torch.tensor(nb, dtype=torch.float32).to(device)
    
    # Align the frames inside each sequence
    warped, tt = torchmfbd.destretch(wb[:, None, :, :, :],
                                     ngrid=16, 
                                     lr=0.20,
                                     reference_frame=0,                                     
                                     border=6,
                                     n_iterations=60,
                                     lambda_tt=0.01)
    
    # Now align all sequences
    warped2 = rearrange(warped, 'ns no nf nx ny -> nf no ns nx ny')    
    # warped_seq = torchmfbd.align_sequence(warped2[0, 0, ...],
    #                                  lr=0.005,
    #                                  border=0,
    #                                  region=None,
    #                                  n_iterations=150,              
    #                                  mode='bilinear',
    #                                  padding_mode='zeros',
    #                                  no_shear=True)

    
    warped_seq, tt_seq = torchmfbd.destretch(warped2[0:1, ...],
                                     ngrid=1, 
                                     lr=0.20,
                                     reference_frame='avg',                                     
                                     border=6,
                                     n_iterations=60,
                                     lambda_tt=0.01)
    # Add both tiptilts
    tt_final = tt + tt_seq[0, :, None, ...]

    # Apply it to the original frames
    tt_final = rearrange(tt_final, 'nb nf d nx ny -> (nb nf) d nx ny')
    wb_final = torchmfbd.apply_destretch(wb[:, None, :, :, :], tt_final).cpu().numpy()
    nb_final = torchmfbd.apply_destretch(nb[:, None, :, :, :], tt_final).cpu().numpy()


    # # ns, no, nf, nx, ny
    im = np.concatenate([wb_final, nb_final], axis=1)
    im_d = None

    print(f"Saving observations to {obs_file}...")
    f = h5py.File(obs_file, 'w')
    f.create_dataset('im', data=im)
    f.close()

    # Save movie: use 3rd axis (nf) as frames and concatenate all sequence frames (ns)
    movie_file = os.path.splitext(obs_file)[0] + '.mp4'
    movie_cube = rearrange(im, 'ns no nf nx ny -> (ns nf) no nx ny')

    # If multiple channels exist in axis=1, display them side-by-side
    if movie_cube.shape[1] > 1:
        movie_frames = np.concatenate([movie_cube[:, i, :, :] for i in range(movie_cube.shape[1])], axis=-1)
    else:
        movie_frames = movie_cube[:, 0, :, :]

    fig, ax = pl.subplots(figsize=(8, 8))
    vmin, vmax = np.percentile(movie_frames[0], [1, 99])
    img = ax.imshow(movie_frames[0], cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title('Aligned sequence movie')
    ax.set_axis_off()

    print(f"Saving movie to {movie_file}...")
    writer = animation.FFMpegWriter(fps=20)
    with writer.saving(fig, movie_file, dpi=150):
        for k in range(movie_frames.shape[0]):
            img.set_data(movie_frames[k])
            ax.set_title(f'Aligned sequence movie - frame {k+1}/{movie_frames.shape[0]}')
            writer.grab_frame()
    pl.close(fig)

    warped_seq = warped_seq[0, 0, ...].cpu().numpy()
    movie_file = os.path.splitext(obs_file)[0] + '_align.mp4'
    fig, ax = pl.subplots(figsize=(8, 8))
    vmin, vmax = np.percentile(warped_seq[0, :, :], [1, 99])
    img = ax.imshow(warped_seq[0, :, :], cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title('Aligned sequence movie')
    ax.set_axis_off()

    print(f"Saving movie to {movie_file}...")
    writer = animation.FFMpegWriter(fps=20)
    with writer.saving(fig, movie_file, dpi=150):
        for k in range(warped_seq.shape[0]):
            img.set_data(warped_seq[k, :, :])
            ax.set_title(f'Aligned sequence movie - frame {k+1}/{warped_seq.shape[0]}')
            writer.grab_frame()
    pl.close(fig)

    