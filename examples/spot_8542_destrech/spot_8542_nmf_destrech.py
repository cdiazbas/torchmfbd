"""Destretch the burst of ../spot_8542 to its average geometry and save it. To deconvolve the result with the
reference example (from ../spot_8542):

    python spot_8542_kl_patches.py ../spot_8542_destrech/spot_20200727_083509_8542_npix512_destretched.h5
"""
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as pl
import os

import torchmfbd

if __name__ == '__main__':
    # Configuration
    obs_file = "../spot_8542/spot_20200727_083509_8542_npix512_original.h5"

    print(f'Reading observations from {obs_file}...')
    with h5py.File(obs_file, 'r') as f:
        im = f['im'][:]    

    # Use the full image size from the file
    n_scans, n_obj, n_frames, nx, ny = im.shape
    means = np.mean(im, axis=(-1, -2), keepdims=True)
    frames = im / means
    frames = torch.tensor(frames.astype('float32'))

    # Calculate contrast to find reference frame
    # frames shape: (n_scans, n_obj, n_frames, nx, ny)
    contrast = torch.std(frames, dim=(-1, -2)) / torch.mean(frames, dim=(-1, -2))
    ind_best_contrast = torch.argmax(contrast[0, 0, :])
    print(f"Reference frame index: {ind_best_contrast}")

    # --- Destretching ---
    print("Performing Destretching to Average Geometry...")
    # We destretch the first object
    warped, tt = torchmfbd.destretch(frames[:, 0:1, ...],
                                    ngrid=32, 
                                    lr=0.2,
                                    reference_frame=ind_best_contrast,
                                    border=6,
                                    n_iterations=100,
                                    lambda_tt=0.01,
                                    average_geometry=True,
                                    n_refine=2)
    
    # Also apply the same flow to the second object (if it exists)
    if frames.shape[1] > 1:
        print("Applying same flows to second object...")
        warped_obj2 = torchmfbd.apply_destretch(frames[:, 1:2, ...], tt)
        warped = torch.cat([warped, warped_obj2], dim=1)

    print("Destretching complete.")

    # --- Metrics ---
    # Temporal RMS (Standard Deviation over time)
    # Lower is better (indicates less jitter)
    rms_orig = torch.std(frames[0, 0, ...], dim=0).mean().item()
    rms_warped = torch.std(warped[0, 0, ...], dim=0).mean().item()
    improvement = (rms_orig - rms_warped) / rms_orig * 100
    print(f"Temporal RMS (Original): {rms_orig:.6f}")
    print(f"Temporal RMS (Warped):   {rms_warped:.6f}")
    print(f"Stability Improvement:    {improvement:.2f}%")

    # --- Save Destretched Data ---
    output_h5 = "spot_20200727_083509_8542_npix512_destretched.h5"
    print(f"Saving destretched frames to {output_h5}...")
    with h5py.File(output_h5, 'w') as f:
        # warped shape is (nb, no, nf, nx, ny)
        # Restore original intensity
        warped_unnorm = warped.cpu().numpy() * means
        f.create_dataset('im', data=warped_unnorm, compression="gzip")
    print("Saving complete.")

    # --- Plotting Results ---
    fig, ax = pl.subplots(2, 2, figsize=(12, 10))
    
    # Original vs Warped for Frame 0 (usually shows the most shift if reference is different)
    ax[0, 0].imshow(frames[0, 0, 0, :, :], cmap='gray')
    ax[0, 0].set_title('Original Frame 0')
    
    ax[0, 1].imshow(warped[0, 0, 0, :, :], cmap='gray')
    ax[0, 1].set_title(f'Destretched (RMS: {rms_warped:.4f}, Improv: {improvement:.1f}%)')

    # Show the flow field (tip-tilt) for Frame 0
    # tt shape: (nb*nf, 2, nx, ny)
    flow_x = tt[0, 0, :, :].cpu().numpy()
    flow_y = tt[0, 1, :, :].cpu().numpy()
    
    im_flow = ax[1, 0].imshow(np.sqrt(flow_x**2 + flow_y**2))
    ax[1, 0].set_title('Flow Magnitude (Frame 0)')
    fig.colorbar(im_flow, ax=ax[1, 0])

    # Show difference for Frame 0
    diff = (warped[0, 0, 0, :, :] - frames[0, 0, 0, :, :]).cpu().numpy()
    ax[1, 1].imshow(diff, cmap='bwr')
    ax[1, 1].set_title('Difference (Warped - Original)')

    pl.tight_layout()
    plot_file = 'destretch_comparison.png'
    pl.savefig(plot_file, dpi=150)
    print(f"Comparison plot saved to {plot_file}")

    # --- Save Movie (Optional, but user requested) ---
    # We can use matplotlib animation to show the sequence before and after
    import matplotlib.animation as animation

    print("Generating comparison movie...")
    fig_anim, (ax1, ax2) = pl.subplots(1, 2, figsize=(10, 5))
    
    # Pre-calculate vmin/vmax for stable colors
    vmn, vmx = frames[0, 0].min(), frames[0, 0].max()
    
    ims = []
    for i in range(frames.shape[2]):
        im1 = ax1.imshow(frames[0, 0, i, :, :], cmap='gray', animated=True, vmin=vmn, vmax=vmx)
        im2 = ax2.imshow(warped[0, 0, i, :, :], cmap='gray', animated=True, vmin=vmn, vmax=vmx)
        if i == 0:
            ax1.imshow(frames[0, 0, i, :, :], cmap='gray', vmin=vmn, vmax=vmx)
            ax2.imshow(warped[0, 0, i, :, :], cmap='gray', vmin=vmn, vmax=vmx)
        ims.append([im1, im2])

    ax1.set_title('Original Sequence')
    ax2.set_title('Destretched (Avg Geometry)')
    
    ani = animation.ArtistAnimation(fig_anim, ims, interval=100, blit=True)
    fig_anim.tight_layout()
    movie_file = 'destretch_movie.mp4'
    try:
        ani.save(movie_file, writer='ffmpeg', savefig_kwargs={'bbox_inches': 'tight'})
        print(f"Movie saved to {movie_file}")
    except Exception as e:
        print(f"Could not save movie (ffmpeg might be missing): {e}")
        # Save as gif as fallback
        try:
            ani.save('destretch_movie.gif', writer='pillow', savefig_kwargs={'bbox_inches': 'tight'})
            print("Movie saved as destretch_movie.gif")
        except:
            print("Could not save animation.")

    pl.close(fig_anim)
