"""Coupling of the wavefronts of neighbouring patches (regularization `coupling` with Deconvolution.set_neighbours)."""
from pathlib import Path
import copy
import numpy as np
import pytest
import torch
import yaml
import torchmfbd

CONFIG = Path(__file__).parent / 'joint.yaml'


def _config(coupling=None):
    with open(CONFIG) as f:
        config = yaml.safe_load(f)
    config['optimization']['gpu'] = -1
    config['regularization'] = {} if coupling is None else {'coupling': coupling}
    return config


def test_neighbours_one_stride_apart():
    # Patch centres from patchifying coordinate images: neighbours are exactly one stride apart in x or y
    n, ps, st = 160, 64, 32
    p4 = torchmfbd.Patchify4D()
    cy = p4.patchify(torch.arange(n, dtype=torch.float32)[:, None].expand(n, n)[None, None].contiguous(), patch_size=ps, stride_size=st)[:, 0]
    cx = p4.patchify(torch.arange(n, dtype=torch.float32)[None, :].expand(n, n)[None, None].contiguous(), patch_size=ps, stride_size=st)[:, 0]
    cen = np.array([(float(cy[p].mean()), float(cx[p].mean())) for p in range(cy.shape[0])])
    d = np.hypot(*(cen[:, None, :] - cen[None, :, :]).transpose(2, 0, 1))
    expected = sorted((p, q) for p in range(len(cen)) for q in range(p + 1, len(cen)) if abs(d[p, q] - st) < 0.5)
    assert sorted(p4.neighbours()) == expected


def test_neighbours_of_several_images_are_not_mixed():
    p4 = torchmfbd.Patchify4D()
    p4.patchify(torch.zeros(2, 3, 128, 128), patch_size=64, stride_size=32)      # 2 images x 9 patches
    pairs = p4.neighbours()
    assert len(pairs) == 2 * 12
    assert all((p < 9) == (q < 9) for p, q in pairs)


def _regularized(coupling):
    d = torchmfbd.Deconvolution(_config(coupling))
    d.set_regularizations()                    # deconvolve() calls it; the regularization block is read there
    return d


def test_config():
    d = _regularized({'lambda': 30.0, 'n_modes': 7})
    assert d.coupling_lambda == 30.0 and d.coupling_modes == 7
    assert _regularized(None).coupling_lambda == 0.0
    with pytest.raises(ValueError):
        _regularized({'lambda': -1.0})


def _fit(make_synthetic_frames, coupling, pairs, n_iterations=5):
    obj, frames = make_synthetic_frames(CONFIG, 4, 5, 0.005, seed=0)
    torch.manual_seed(0)
    d = torchmfbd.Deconvolution(_config(coupling))
    d.add_frames(frames, sigma=0.005 * torch.ones(4, 5), id_object=0, id_diversity=0, diversity=0.0)
    if pairs is not None:
        d.set_neighbours(pairs)
    d.deconvolve(infer_object=False, optimizer='lbfgs', simultaneous_sequences=200, n_iterations=n_iterations)
    return d


def test_zero_lambda_is_bit_identical(make_synthetic_frames):
    ref = _fit(make_synthetic_frames, None, None)
    d = _fit(make_synthetic_frames, {'lambda': 0.0}, [(0, 1), (1, 2), (2, 3)])
    assert torch.equal(ref.obj[0], d.obj[0])
    assert torch.equal(ref.modes_seq[0], d.modes_seq[0])


def test_strong_coupling_pulls_neighbours_together(make_synthetic_frames):
    pairs = [(0, 1), (1, 2), (2, 3)]
    free = _fit(make_synthetic_frames, None, None)
    tied = _fit(make_synthetic_frames, {'lambda': 1e6, 'n_modes': 10}, pairs)
    def spread(d):
        m = d.modes_seq[0][..., 2:10]                           # coupled modes beyond tip-tilt
        return float(sum(torch.sum((m[p] - m[q])**2) for p, q in pairs))
    assert spread(tied) < 0.01 * spread(free)


def test_unpaired_batches_are_reported(make_synthetic_frames, caplog):
    obj, frames = make_synthetic_frames(CONFIG, 4, 5, 0.005, seed=0)
    d = torchmfbd.Deconvolution(_config({'lambda': 30.0}))
    d.add_frames(frames, sigma=0.005 * torch.ones(4, 5), id_object=0, id_diversity=0, diversity=0.0)
    d.set_neighbours([(0, 1), (1, 2), (2, 3)])
    d.deconvolve(infer_object=False, optimizer='lbfgs', simultaneous_sequences=2, n_iterations=1)
    assert any('different batches' in r.getMessage() for r in caplog.records)
