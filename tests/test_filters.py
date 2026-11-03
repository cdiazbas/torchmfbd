import torchmfbd
import torch
import numpy as np
import yaml
from skimage.segmentation import flood
from pathlib import Path

def _config(image_filter):

    config_path = Path(__file__).parent / 'joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['object1']['image_filter'] = image_filter

    return config_path, config

def _deconvolve(config, frames, sigma, n_iterations):

    n_seq, n_frames = frames.shape[0:2]

    torch.manual_seed(0)
    decSI = torchmfbd.Deconvolution(config)
    decSI.add_frames(frames, sigma=sigma * torch.ones(n_seq, n_frames), id_object=0, id_diversity=0, diversity=0.0)
    decSI.deconvolve(infer_object=False,
                     optimizer='adam',
                     simultaneous_sequences=200,
                     n_iterations=n_iterations)

    return decSI

def test_garrote_gain(make_synthetic_frames):

    config_path, config = _config('garrote')
    config['optimization']['gpu'] = -1
    obj, frames = make_synthetic_frames(config_path, 2, 5, 0.01, seed=1)

    dec = torchmfbd.Deconvolution(config)
    dec.n_o = 1
    dec.init_frame_diversity = [[0]]
    dec._define_basis()

    modes = 0.5 * torch.randn(2, 5, dec.n_modes, generator=torch.Generator().manual_seed(2)) / torch.sqrt(torch.arange(dec.n_modes) + 1.0)
    _, otf = dec.compute_psfs(modes, [torch.zeros(2, 5)])
    images_ft = torch.fft.fft2(frames, norm='ortho')

    Sconj_S = torch.sum(torch.conj(otf[0]) * otf[0], dim=1)
    Sconj_I = torch.sum(torch.conj(otf[0]) * images_ft, dim=1)
    s2 = torch.full((2,), 0.01**2)

    gain = dec.garrote_filter(Sconj_S, Sconj_I, s2)

    assert gain.shape == Sconj_S.shape, f'Expected {Sconj_S.shape}, got {gain.shape}'
    assert torch.all((gain >= 0.0) & (gain <= 1.0)), f'Gain outside [0, 1]: [{gain.min().item()}, {gain.max().item()}]'
    assert torch.all(gain[:, 0, 0] > 0.99), f'Gain at zero frequency {gain[:, 0, 0].tolist()}'

    # No gain where the OTFs vanish (their values there are roundoff)
    outside = Sconj_S.real <= 1e-10 * Sconj_S.real[:, 0:1, 0:1]
    assert torch.all(gain[outside] == 0.0), 'Non-zero gain where the OTFs vanish'

    # A change of the data at the 1e-7 level changes the gain at that level, not by a jump
    Sconj_I_pert = Sconj_I * (1.0 + 1e-7 * torch.randn(Sconj_I.shape, generator=torch.Generator().manual_seed(3)))
    change = torch.max(torch.abs(dec.garrote_filter(Sconj_S, Sconj_I_pert, s2) - gain))
    assert change < 1e-4, f'Gain changed by {change.item()} under a 1e-7 change of the data'

    # The reconstruction thresholded at a level is the flood fill from zero frequency of the gain above it
    raw = dec.garrote_filter.__func__
    dec.reconstruct_from_zero_frequency = lambda g: g
    gain_raw = raw(dec, Sconj_S, Sconj_I, s2)
    del dec.reconstruct_from_zero_frequency
    for level in (0.2, 0.5):
        for b in range(gain.shape[0]):
            g_shift = torch.fft.fftshift(gain_raw[b]).numpy() >= level
            ff = flood(g_shift.astype(np.uint8), (16, 16), footprint=np.ones((3, 3)), tolerance=0)
            rec = torch.fft.fftshift(gain[b]).numpy() >= level
            assert np.array_equal(rec, ff), f'Reconstruction above {level} differs from the flood fill in {int((rec != ff).sum())} frequencies'

def test_reconstruction():

    config_path, config = _config('garrote')
    config['optimization']['gpu'] = -1
    dec = torchmfbd.Deconvolution(config)

    # Gain 0.9 in a disc around zero frequency and in an island separated from it, 0 elsewhere
    n = 32
    k = torch.fft.fftfreq(n) * n
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    disc = (kx**2 + ky**2) <= 5**2
    island = ((kx - 11)**2 + (ky - 3)**2) <= 2**2
    gain = 0.9 * (disc | island).float()[None]

    rec = dec.reconstruct_from_zero_frequency(gain)
    assert torch.equal(rec[0][disc], gain[0][disc]), 'Gain connected to zero frequency was changed'
    assert torch.all(rec[0][island] == 0.0), 'Island not connected to zero frequency was kept'

    # A narrow gap limits what lies behind it to the gain of the gap
    gain[0][island] = 0.9
    bridge = (kx >= 5) & (kx <= 9) & (ky == 3)
    gain[0][bridge] = 0.3
    rec = dec.reconstruct_from_zero_frequency(gain)
    assert torch.allclose(rec[0][island], torch.tensor(0.3)), f'Gain behind a 0.3 gap is {rec[0][island].unique().tolist()}'

def test_garrote_object(make_synthetic_frames, object_error):

    n_seq = 4
    n_frames = 5
    sigma = 0.005

    config_path, config = _config('garrote')
    obj, frames = make_synthetic_frames(config_path, n_seq, n_frames, sigma, seed=0)

    decSI = _deconvolve(config, frames, sigma, 100)

    assert decSI.obj[0].shape == obj.shape, f'Expected {obj.shape}, got {decSI.obj[0].shape}'
    assert torch.all(torch.isfinite(decSI.obj[0])), 'Non-finite object'

    err_obj, err_frame = object_error(decSI, obj, frames)
    ratio = torch.mean(err_obj) / torch.mean(err_frame)
    assert ratio < 0.75, f'Mean object error is {ratio.item()} of the mean best frame error'

def test_garrote_continuous(make_synthetic_frames):

    n_seq = 2
    n_frames = 5
    sigma = 0.01

    config_path, config = _config('garrote')
    obj, frames = make_synthetic_frames(config_path, n_seq, n_frames, sigma, seed=0)

    # Six steps: the modes of the two runs still agree to roundoff, so what is measured is the filter
    pert = 1.0 + 1e-7 * torch.randn(frames.shape, generator=torch.Generator().manual_seed(9))
    out = [_deconvolve(config, f, sigma, 6).obj[0].cpu() for f in (frames, frames * pert)]

    change = torch.sqrt(torch.mean((out[1] - out[0])**2)) / torch.std(out[0])
    assert change < 1e-3, f'Object changed by {change.item()} of its std under a 1e-7 change of the frames'
