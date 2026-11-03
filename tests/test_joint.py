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
