import copy
import numpy as np
import torch
import yaml
import torchmfbd
from pathlib import Path


def _config(half):
    with open(Path(__file__).parent / 'joint.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['optimization']['half_spectrum'] = half
    config['optimization']['gpu'] = -1
    return config


def _fitted(make_synthetic_frames, half, n_iterations):
    n_seq, n_frames, sigma = 3, 5, 0.005
    obj, frames = make_synthetic_frames(Path(__file__).parent / 'joint.yaml', n_seq, n_frames, sigma, seed=0)
    dec = torchmfbd.Deconvolution(_config(half))
    dec.add_frames(frames, sigma=sigma * torch.ones(n_seq, n_frames), id_object=0, id_diversity=0, diversity=0.0)
    torch.manual_seed(0)
    dec.deconvolve(infer_object=False, optimizer='adam', simultaneous_sequences=200, n_iterations=n_iterations)
    return dec


def test_half_spectrum_is_the_same_loss(make_synthetic_frames):
    """The half-plane loss has the value and the gradient of the full one, at any wavefront."""
    dec = _fitted(make_synthetic_frames, False, 1)
    frames_ft = [dec.frames_ft[0].to(dec.device)]
    sigma = [dec.sigma[0].to(dec.device)]
    div = [dec.diversity[0].to(dec.device)]
    g = torch.Generator().manual_seed(1)
    for n_active in (2, 10, 30):
        m = (0.3 * torch.randn(frames_ft[0].shape[0], frames_ft[0].shape[1], n_active, generator=g)).requires_grad_(True)
        _, otf = dec.compute_psfs(m, div, half=False)
        L = dec.compute_loss(frames_ft, otf, sigma)[2]
        gF = torch.autograd.grad(L, m)[0]
        _, otf_half = dec.compute_psfs(m, div, half=True)
        Lh = dec.compute_loss_half(frames_ft, otf_half, sigma)[2]
        gH = torch.autograd.grad(Lh, m)[0]
        assert abs(float(L - Lh)) < 1e-5 * abs(float(L)), f'{n_active} modes: loss {float(L)} vs half {float(Lh)}'
        assert float((gF - gH).norm()) < 1e-4 * float(gF.norm()), f'{n_active} modes: gradients differ'


def test_half_spectrum_fit(make_synthetic_frames):
    """A short fit with half_spectrum gives the object of the full-spectrum fit, to rounding."""
    full = _fitted(make_synthetic_frames, False, 10)
    half = _fitted(make_synthetic_frames, True, 10)
    a, b = full.obj[0].cpu(), half.obj[0].cpu()
    assert torch.max(torch.abs(a - b)) < 1e-3 * torch.max(torch.abs(a)), f'objects differ by {float(torch.max(torch.abs(a - b)))}'
