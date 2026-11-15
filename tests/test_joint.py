import torchmfbd
import torch
import yaml
from pathlib import Path

def test_joint(make_synthetic_frames, object_error):

    n_seq = 4
    n_frames = 5
    sigma = 0.005

    config_path = Path(__file__).parent / 'joint.yaml'

    obj, frames = make_synthetic_frames(config_path, n_seq, n_frames, sigma, seed=0)

    decSI = torchmfbd.Deconvolution(config_path)
    decSI.add_frames(frames, sigma=sigma * torch.ones(n_seq, n_frames), id_object=0, id_diversity=0, diversity=0.0)

    decSI.deconvolve(infer_object=False,
                     optimizer='adam',
                     simultaneous_sequences=200,
                     n_iterations=100)

    loss = decSI.loss
    assert torch.all(torch.isfinite(loss)), f'Non-finite loss at iterations {torch.where(~torch.isfinite(loss))[0].tolist()}'
    assert loss[-1] < loss[0], f'Loss did not decrease: {loss[0].item()} -> {loss[-1].item()}'

    assert decSI.obj[0].shape == obj.shape, f'Expected {obj.shape}, got {decSI.obj[0].shape}'
    assert torch.all(torch.isfinite(decSI.obj[0])), 'Non-finite object'

    err_obj, err_frame = object_error(decSI, obj, frames)
    assert torch.all(err_obj < err_frame), f'Object error {err_obj.tolist()} not below best frame error {err_frame.tolist()}'

    # Averaged over the sequences, 0.75 lies between the error with the fitted PSFs and the error with
    # the diffraction-limited PSF, so the check also fails if the wavefront is not fitted
    ratio = torch.mean(err_obj) / torch.mean(err_frame)
    assert ratio < 0.75, f'Mean object error is {ratio.item()} of the mean best frame error'

def _time_config():

    config_path = Path(__file__).parent / 'joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['optimization']['filter_loss'] = 'wiener_time'
    config['regularization'] = {'time': {'lambda': 0.0}}

    return config_path, config

def test_time_4d(make_synthetic_frames):

    n_seq = 2
    n_frames = 5
    sigma = 0.005

    config_path, config = _time_config()

    obj, frames = make_synthetic_frames(config_path, n_seq, n_frames, sigma, seed=0)

    # A single time step per sequence (n_t = 1)
    decSI = torchmfbd.Deconvolution(config)
    decSI.add_frames(frames, sigma=sigma * torch.ones(n_seq, n_frames), id_object=0, id_diversity=0, diversity=0.0)

    decSI.deconvolve(infer_object=False,
                     optimizer='adam',
                     simultaneous_sequences=200,
                     n_iterations=20)

    assert torch.all(torch.isfinite(decSI.loss)), f'Non-finite loss at iterations {torch.where(~torch.isfinite(decSI.loss))[0].tolist()}'

    out = decSI.obj[0].cpu()
    assert out.shape == (n_seq, 32, 32), f'Expected {(n_seq, 32, 32)}, got {out.shape}'
    assert out.dtype == torch.float32, f'Expected float32, got {out.dtype}'
    assert torch.all(torch.isfinite(out)), 'Non-finite object'

    # The object keeps the mean level of the frames
    assert torch.allclose(out.mean(dim=(-1, -2)), frames.mean(dim=(-1, -2, -3)), rtol=0.05), f'Object mean {out.mean(dim=(-1, -2)).tolist()}, frames mean {frames.mean(dim=(-1, -2, -3)).tolist()}'

def test_time_5d(make_synthetic_frames):

    n_seq = 2
    n_frames = 5
    sigma = 0.005

    config_path, config = _time_config()

    obj, frames = make_synthetic_frames(config_path, n_seq, n_frames, sigma, seed=0)

    # The two sequences as two time steps of one sequence, (s, t, f, x, y)
    decSI = torchmfbd.Deconvolution(config)
    decSI.add_frames(frames[None, ...], sigma=sigma * torch.ones(1, n_seq, n_frames), id_object=0, id_diversity=0, diversity=0.0)

    decSI.deconvolve(infer_object=False,
                     optimizer='adam',
                     simultaneous_sequences=200,
                     n_iterations=20)

    assert torch.all(torch.isfinite(decSI.loss)), f'Non-finite loss at iterations {torch.where(~torch.isfinite(decSI.loss))[0].tolist()}'

    out = decSI.obj[0].cpu()
    assert out.shape == (1, n_seq, 32, 32), f'Expected {(1, n_seq, 32, 32)}, got {out.shape}'
    assert out.dtype == torch.float32, f'Expected float32, got {out.dtype}'
    assert torch.all(torch.isfinite(out)), 'Non-finite object'
    assert torch.allclose(out.mean(dim=(-1, -2)), frames.mean(dim=(-1, -2, -3))[None, :], rtol=0.05), f'Object mean {out.mean(dim=(-1, -2)).tolist()}, frames mean {frames.mean(dim=(-1, -2, -3)).tolist()}'

def test_prior_frames_scale():

    # The joint object prior keeps its weight relative to the data above 12 frames (s_u_joint * 12 / N) and is
    # unchanged at 12 frames or fewer; s_u_joint is stated for 64-px patches (here 32 px: factor (32/64)^2)
    config_path = Path(__file__).parent / 'joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['optimization']['gpu'] = -1
    config['images']['remove_fixed_pattern'] = False

    s_u_joint = config['object1']['s_u_joint']
    for n_frames, expected in ((5, 1.0), (12, 1.0), (24, 0.5), (48, 0.25)):
        dec = torchmfbd.Deconvolution(config)
        dec.add_frames(1.0 + 0.01 * torch.rand(1, n_frames, 32, 32), sigma=0.01, id_object=0, id_diversity=0, diversity=0.0)
        dec.n_o = 1
        dec.init_frame_diversity = [[0]]
        dec._define_basis()
        dec.combine_frames()
        s_u, _, _, _, _ = dec.get_su_s2(obj=0, sigma=[0.01 * torch.ones(1, n_frames)])
        value = float(s_u.flatten()[1])
        assert abs(value - s_u_joint * 0.25 * expected) < 1e-6 * s_u_joint, f'{n_frames} frames: s_u = {value}, expected {s_u_joint * 0.25 * expected}'

def test_object_independent_of_batch_noise():

    # A sequence's object must not depend on the noise of the other sequences deconvolved in the same batch:
    # sequence A (sigma 0.01) alone and together with a noisy sequence B (sigma 0.1), true OTFs (BUG-170)
    config_path = Path(__file__).parent / 'joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['optimization']['gpu'] = -1
    config['images']['remove_fixed_pattern'] = False

    gen = torch.Generator().manual_seed(3)
    n_frames = 6
    for image_filter in ('scharmer', 'garrote', 'tophat'):
        dec = torchmfbd.Deconvolution(config)
        dec.n_o = 1
        dec.n_t = 1
        dec.init_frame_diversity = [[0]]
        dec.image_filter = [image_filter]
        dec.loss_filter = 'wiener'
        dec._define_basis()
        dec.compute_diffraction_masks()
        npix = dec.npix

        obj = 1.0 + 0.1 * torch.randn(2, npix, npix, generator=gen)
        modes = 0.4 * torch.randn(2, n_frames, dec.n_modes, generator=gen) / torch.sqrt(torch.arange(dec.n_modes) + 1.0)
        modes[:, :, 0:2] = 0.0
        with torch.no_grad():
            _, otf = dec.compute_psfs(modes, [torch.zeros(2, n_frames)])
        sigma = torch.tensor([[0.01] * n_frames, [0.1] * n_frames])
        noise = torch.randn(2, n_frames, npix, npix, generator=gen) * sigma[:, :, None, None]
        frames_ft = npix * torch.fft.fft2(obj, norm='ortho')[:, None] * otf[0] + torch.fft.fft2(noise, norm='ortho')

        with torch.no_grad():
            _, _, both = dec.compute_object([frames_ft], [otf[0]], [sigma], None)
            _, _, alone = dec.compute_object([frames_ft[0:1]], [otf[0][0:1]], [sigma[0:1]], None)

        diff = float(torch.max(torch.abs(both[0][0] - alone[0][0])) / torch.std(alone[0][0]))
        assert diff < 1e-4, f'{image_filter}: the object of sequence A changes by {diff:.3e} std when a noisy sequence shares its batch'

def test_jitter_with_threaded_fft(make_synthetic_frames):

    # The jitter OTF multiplies the OTF returned by the threaded FFT, a custom autograd function: that product
    # must not be in place (BUG-172). The threaded FFT is used on CPU builds without MKL when the batch has
    # at least 4 images per thread, so two threads and 2 x 5 frames reach it
    config_path = Path(__file__).parent / 'joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['optimization']['gpu'] = -1
    config['images']['remove_fixed_pattern'] = False
    config['annealing']['type'] = 'none'

    obj, frames = make_synthetic_frames(config_path, 2, 5, 0.005, seed=0)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for jitter in ('slow', 'fast'):
            config['psf']['jitter'] = jitter
            dec = torchmfbd.Deconvolution(config)
            dec.add_frames(frames, sigma=0.005 * torch.ones(2, 5), id_object=0, id_diversity=0, diversity=0.0)
            dec.deconvolve(infer_object=False, optimizer='adam', simultaneous_sequences=2, n_iterations=2)
            assert torch.all(torch.isfinite(dec.obj[0])), f'jitter {jitter}: non-finite object'
    finally:
        torch.set_num_threads(n_threads)
