import glob
import numpy as np
import matplotlib.pyplot as pl
import h5py
import matplotlib.animation as animation

# Rows: the unregularized reconstruction and every regularized one that spot_8542_joint.py has written
f = h5py.File('reconstruction_original.h5', 'r')
n_t = f['obj_0'].shape[0]
orig_0 = f['obj_0'][:]
orig_1 = f['obj_1'][:]
best_0 = f['best_frame_0'][:]
best_1 = f['best_frame_1'][:]
f.close()

titles = ['No regularization']
regu0 = [orig_0]
regu1 = [orig_1]
for filename in sorted(glob.glob('reconstruction_regularized_*.h5')):
    f = h5py.File(filename, 'r')
    regu0.append(f['obj_0'][:])
    regu1.append(f['obj_1'][:])
    f.close()
    titles.append('Regularized (' + filename[len('reconstruction_regularized_'):-len('.h5')].replace('_', ' ') + ')')

n_rows = len(titles)
im = [None] * (4 * n_rows)

fig, ax = pl.subplots(ncols=4, nrows=n_rows, figsize=(10, 2.5 * n_rows), sharex=True, sharey=True, squeeze=False)

for j in range(n_rows):
    im[4*j] = ax[j, 0].imshow(best_0[0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
    im[4*j+1] = ax[j, 1].imshow(regu0[j][0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
    im[4*j+2] = ax[j, 2].imshow(best_1[0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
    im[4*j+3] = ax[j, 3].imshow(regu1[j][0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
    ax[j, 0].set_title(titles[j])

def update(frame):
    for j in range(n_rows):
        im[4*j].set_array(best_0[frame, 0, :, :])
        im[4*j+1].set_array(regu0[j][frame, 0, :, :])
        im[4*j+2].set_array(best_1[frame, 0, :, :])
        im[4*j+3].set_array(regu1[j][frame, 0, :, :])

    return im

ani = animation.FuncAnimation(fig, 
                                  update, 
                                  frames=n_t, 
                                  interval=100, 
                                  blit=True)

# Save the animation (needs ffmpeg)
output_filename = f"movie.mp4"
writer = animation.FFMpegWriter(fps=6, metadata=dict(artist='Me'), bitrate=1800)
ani.save(output_filename, writer=writer)
