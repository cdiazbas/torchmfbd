import torchmfbd
import torch
import yaml
from pathlib import Path

def _run(make_synthetic_frames, nullspace):

    n_seq = 4
    n_frames = 5
    sigma = 0.005

    config_path = Path(__file__).parent / 'joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['psf'].update(nullspace)

    obj, frames = make_synthetic_frames(config_path, n_seq, n_frames, sigma, seed=0)

    dec = torchmfbd.Deconvolution(config)
    dec.add_frames(frames, sigma=sigma * torch.ones(n_seq, n_frames), id_object=0, id_diversity=0, diversity=0.0)
    dec.deconvolve(infer_object=False, optimizer='lbfgs', simultaneous_sequences=200, n_iterations=10)

    return dec, obj, frames

def test_nullspace_keep_all(make_synthetic_frames):

    # A threshold below every eigenvalue keeps all directions: P is the identity and the second pass repeats the first
    plain, _, _ = _run(make_synthetic_frames, {})
    full, _, _ = _run(make_synthetic_frames, {'nullspace_d_r0': 5.0, 'nullspace_threshold': -1.0})

    n = full.n_f * full.n_modes
    assert all(k == n for k in full.nullspace_kept), f'Kept {full.nullspace_kept} of {n} directions'
    for P in full.nullspace_P_seq:
        assert torch.allclose(P, torch.eye(n).expand_as(P), atol=1e-5), 'Projector is not the identity'
    assert torch.allclose(full.obj[0], plain.obj[0], atol=1e-5), \
        f'Object differs from the unrestricted fit by {torch.max(torch.abs(full.obj[0] - plain.obj[0])).item()}'

def test_nullspace_restricted(make_synthetic_frames, object_error):

    dec, obj, frames = _run(make_synthetic_frames, {'nullspace_d_r0': 5.0})

    n = dec.n_f * dec.n_modes
    assert all(0 < k < n for k in dec.nullspace_kept), f'Kept {dec.nullspace_kept} of {n} directions'

    # Oblique projector: P P = P
    for P in dec.nullspace_P_seq:
        err = torch.max(torch.abs(P @ P - P)).item()
        assert err < 1e-3 * torch.max(torch.abs(P)).item(), f'P P differs from P by {err}'

    assert torch.all(torch.isfinite(dec.loss)), 'Non-finite loss'
    assert torch.all(torch.isfinite(dec.obj[0])), 'Non-finite object'

    err_obj, err_frame = object_error(dec, obj, frames)
    assert torch.all(err_obj < err_frame), f'Object error {err_obj.tolist()} not below best frame error {err_frame.tolist()}'
