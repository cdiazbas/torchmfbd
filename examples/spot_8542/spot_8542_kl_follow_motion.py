"""
images.follow_motion on the 8542 burst: every frame of every patch cut again where it moved (MOMFBD's MAX_LOCAL_SHIFT).

The burst itself moves by only 0.14-0.28 px rms between frames, where follow_motion has nothing to correct. To show it at
work, a known motion is added: frame j of both cameras (they share the telescope, so they move together) is cut from the
512-px field at an offset of a few pixels. The burst is then fitted twice with kl.yaml, with the fixed cutout and with
follow_motion, and the cuts follow_motion found are compared with the motion added.

    python spot_8542_kl_follow_motion.py [burst.h5] [motion rms in px, default 3]

On the benchmark (benchmarks/run.py, 55 scenes with a known truth) follow_motion = 5 lowers the object error by 9.4 % at
3 px rms of motion and by 39.2 % at 6 px (40.2 % on hold-out scenes), changes nothing without motion, and costs 4 % of the
fit time. See docs/userguide/patches.rst.
"""
import copy
import sys
import time
import numpy as np
import h5py
import torch
import yaml
import matplotlib.pyplot as pl
import torchmfbd


def fit(config, frames, margin):
    """kl.yaml fit of the two cameras, with patches of 128 px (+ margin on each side), stride 32."""
    patchify = torchmfbd.Patchify4D()
    dec = torchmfbd.Deconvolution(config)
    for i in range(2):
        patches = patchify.patchify(frames[:, i], patch_size=128, stride_size=32, flatten_sequences=True, margin=margin)
        dec.add_frames(patches, id_object=i, id_diversity=0, diversity=0.0)
    dec.set_neighbours(patchify.neighbours())          # for the coupling regularization of kl.yaml
    t0 = time.time()
    dec.deconvolve(infer_object=False, optimizer='lbfgs', simultaneous_sequences=patches.shape[0], n_iterations=10)
    t = time.time() - t0
    obj = [patchify.unpatchify(dec.obj[i], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy() for i in range(2)]
    return dec, obj, t


if __name__ == '__main__':

    obs_file = sys.argv[1] if len(sys.argv) > 1 else "spot_20200727_083509_8542_npix512_original.h5"
    motion = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    m = 5                                              # images.follow_motion: the largest cut, px (MOMFBD's default)

    with h5py.File(obs_file, 'r') as f:
        im = f['im'][:]                                # (scans, cameras, frames, 512, 512)
    n_frames = im.shape[2]

    # The added motion: whole-pixel offsets per frame, the same for both cameras, at most 12 px
    rng = np.random.default_rng(0)
    d = np.clip(np.rint(motion * rng.standard_normal((n_frames, 2))), -12, 12).astype(int)
    d[0] = 0
    c, n = 128, 256                                    # a 256-px crop in the middle of the field
    frames = np.stack([im[:, :, j, c + d[j, 0]:c + d[j, 0] + n, c + d[j, 1]:c + d[j, 1] + n] for j in range(n_frames)], axis=2)
    frames = frames / np.mean(frames, axis=(-1, -2), keepdims=True)
    frames = torch.tensor(frames.astype('float32'))
    print(f'Added motion: {np.sqrt(np.mean(d.astype(float)**2)):.1f} px rms per axis')

    with open('kl.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # The fixed cutout (follow_motion 0, the default)
    config_fixed = copy.deepcopy(config)
    config_fixed['images']['follow_motion'] = 0
    dec_fixed, obj_fixed, t_fixed = fit(config_fixed, frames, margin=0)

    # follow_motion: the patches carry a margin of m px, in which every frame is cut again where it moved. It warns,
    # and fits the central cut, where it cannot be used (time series, several wavelengths, ...)
    config_follow = copy.deepcopy(config)
    config_follow['images']['follow_motion'] = m
    dec_follow, obj_follow, t_follow = fit(config_follow, frames, margin=m)

    # The cuts found, (patches, frames, 2), against the motion added: a frame cut from the field at +d shows the scene moved
    # by -d. Both are known up to a common shift (the cuts are referred to the median of the burst), and the seeing moves
    # every frame by a little more, differently in every patch: a cut within 1 px of the motion added is a right one
    found = dec_follow.motion_shifts.numpy()
    want = np.clip(-d, -m, m)
    diff = found - want[None]
    diff = diff - np.median(diff.reshape(-1, 2), axis=0)
    close = np.all(np.abs(diff) <= 1, axis=-1)
    print(f'Cuts within 1 px of the motion added: {100 * close.mean():.0f} % of (patch, frame); rms difference '
          f'{np.sqrt(np.mean(diff**2)):.2f} px; frames that moved more than {m} px are cut at {m} px')
    print(f'Fit time: {t_fixed:.1f} s with the fixed cutout, {t_follow:.1f} s with follow_motion')

    fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(15, 10))
    for i in range(2):
        vmin, vmax = np.percentile(frames[0, i, 0].numpy(), [0.5, 99.5])
        for j, (img, title) in enumerate([(frames[0, i, 0].numpy(), 'Frame 0'),
                                          (obj_fixed[i][0], f'Fixed cutout ({t_fixed:.1f} s)'),
                                          (obj_follow[i][0], f'follow_motion = {m} ({t_follow:.1f} s)')]):
            ax[i, j].imshow(img, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
            ax[i, j].set_title(f'Camera {i}: {title}')
    fig.suptitle(f'Frames moved by {np.sqrt(np.mean(d.astype(float)**2)):.1f} px rms per axis')
    pl.savefig('spot_8542_kl_follow_motion.png', dpi=150, bbox_inches='tight')
