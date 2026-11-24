import copy
import numpy as np
import pytest
import torch
import yaml
import torchmfbd
from pathlib import Path


def _config(follow=0, refine=False):
    with open(Path(__file__).parent / 'joint.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['optimization']['gpu'] = -1
    config['images']['follow_motion'] = follow
    config['images']['follow_motion_refine'] = refine
    return config


def _basis_only(config):
    dec = torchmfbd.Deconvolution(config)
    dec.n_o = 1
    dec.init_frame_diversity = [[0]]
    dec._define_basis()
    return dec


def test_tilt_calibration_moves_the_image():
    """The tip-tilt C^-1 d moves the image by d pixels (to the coma the KL tip-tilt modes carry): the image is far
    closer to the unmoved one rolled by d than to it rolled by -d or not moved."""
    dec = _basis_only(_config())
    C = dec._tilt_calibration(0)
    n = dec.npix
    f = torch.fft.fftfreq(n)
    obj = torch.fft.ifft2(torch.randn(n, n, generator=torch.Generator().manual_seed(0)) *
                          ((f[:, None]**2 + f[None]**2) < 0.15**2)).real
    for d in [(2, -1), (-3, 1), (1, 3)]:
        a = torch.linalg.solve(C, torch.tensor(d, dtype=torch.float64)).float()
        m = torch.zeros(1, 2, dec.n_modes)
        m[0, 1, 0:2] = a
        psf, _ = dec.compute_psfs(m, [torch.zeros(1, 2)])
        im = torch.fft.ifft2(torch.fft.fft2(obj) * torch.fft.fft2(psf[0][0])).real
        err = lambda s: float((im[1] - torch.roll(im[0], shifts=s, dims=(0, 1))).norm() / im[0].norm())
        assert err(d) < 0.5 * min(err((-d[0], -d[1])), err((0, 0))), f'{d}: {err(d)} {err((-d[0], -d[1]))} {err((0, 0))}'


def test_patchify_margin():
    """With a margin the patches are patch_size + 2 margin across, on the same grid: their centre is the plain patch."""
    x = torch.randn(1, 3, 96, 96, generator=torch.Generator().manual_seed(1))
    P, Q = torchmfbd.Patchify4D(), torchmfbd.Patchify4D()
    a = P.patchify(x, patch_size=32, stride_size=16)
    b = Q.patchify(x, patch_size=32, stride_size=16, margin=4)
    assert b.shape[-1] == 40 and b.shape[0] == a.shape[0]
    assert torch.equal(b[..., 4:36, 4:36], a)


def _moving_burst(make_synthetic_frames, motion, m, kmax=12):
    """Synthetic bursts (periodic objects, so a frame with its margin is the wrapped field), with frame motion[j] (dy, dx)
    px moved by rolling it."""
    n_seq, n_frames, sigma = 2, 5, 0.005
    obj, frames = make_synthetic_frames(Path(__file__).parent / 'joint.yaml', n_seq, n_frames, sigma, seed=0, kmax=kmax)
    big = torch.nn.functional.pad(frames, (m, m, m, m), mode='circular')
    for j, s in enumerate(motion):
        big[:, j] = torch.roll(big[:, j], shifts=s, dims=(-2, -1))
    return obj, frames, big, sigma


def _fit(config, frames, sigma, n_iterations=40, modes_std=None):
    dec = torchmfbd.Deconvolution(config)
    dec.logger.setLevel(50)
    dec.add_frames(frames, sigma=sigma * torch.ones(frames.shape[0], frames.shape[1]), id_object=0, id_diversity=0, diversity=0.0)
    torch.manual_seed(0)
    dec.deconvolve(infer_object=False, optimizer='adam', simultaneous_sequences=200, n_iterations=n_iterations)
    return dec


def _shifted(x, s):
    """x moved by the (sub-pixel) shift s = (dy, dx) px, as torch.roll moves it by whole pixels."""
    f = torch.fft.fftfreq(x.shape[-1])
    ramp = torch.exp(-2j * np.pi * (f[:, None] * s[0] + f[None, :] * s[1]))
    return torch.fft.ifft2(torch.fft.fft2(x) * ramp).real


def test_follow_motion_finds_and_undoes_the_motion(make_synthetic_frames, object_error):
    """One frame of five moved by (3, -2) px: follow_motion cuts it back where it was. The correlation alone finds (2, -2)
    in the second burst; follow_motion_refine corrects it from the fitted tip-tilt, so both bursts record (3, -2). The
    object is as close to the truth as the one fitted from frames that did not move; it sits where the mean of the frames
    is (MOMFBD's reference): (0.6, -0.4) px from the truth."""
    m = 4
    motion = [(0, 0), (0, 0), (3, -2), (0, 0), (0, 0)]
    obj, frames, big, sigma = _moving_burst(make_synthetic_frames, motion, m)
    one = _fit(_config(m), big, sigma)
    assert torch.equal(one.motion_shifts[0, 2], torch.tensor([3, -2])), one.motion_shifts
    dec = _fit(_config(m, refine=True), big, sigma)
    want = torch.tensor([[0, 0], [0, 0], [3, -2], [0, 0], [0, 0]])
    assert torch.equal(dec.motion_shifts, want[None].expand(2, 5, 2)), dec.motion_shifts
    ref = _fit(_config(0), frames, sigma)
    e_ref, e_frame = object_error(ref, obj, frames)
    e_fm, _ = object_error(dec, _shifted(obj, (0.6, -0.4)), frames)
    assert torch.all(e_fm < 1.2 * e_ref) and torch.all(e_fm < 0.5 * e_frame), (e_fm, e_ref, e_frame)


def test_follow_motion_keeps_the_total_tilt(make_synthetic_frames):
    """The cut moves a frame by whole pixels and its tip-tilt by the same amount the other way: the tip-tilt the PSFs use
    (coefficient plus offset, referred to the mean) is the same before and after the cut."""
    m = 4
    obj, frames, big, sigma = _moving_burst(make_synthetic_frames, [(0, 0)] * 5, m)
    dec = _fit(_config(m), big, sigma, n_iterations=5)
    C = dec._tilt_calibration(0)
    g = torch.Generator().manual_seed(3)
    modes = dec.modes_seq[0].clone()
    modes[..., 0:2] = (torch.randn(2, 5, 2, generator=g, dtype=torch.float64) * 2.5) @ torch.linalg.inv(C).T
    dec.modes_seq[0] = modes.float()
    before = dec._center_tilt(dec.modes_seq[0].double())[..., 0:2] + dec.tilt_offset.double()
    start = dec._recut_from_tilt()
    assert start is not None and bool((dec.motion_shifts != 0).any())
    dec.tilt_offset = (dec.motion_shifts.double() @ torch.linalg.inv(C).T).float()
    after = dec._center_tilt(start.double())[..., 0:2] + dec.tilt_offset.double()
    assert torch.allclose(before, after, atol=1e-4), float((before - after).abs().max())


def test_follow_motion_refuses():
    """Frames without the margin, and objects at different wavelengths."""
    dec = torchmfbd.Deconvolution(_config(4))
    with pytest.raises(ValueError, match='follow_motion'):
        dec.add_frames(torch.zeros(1, 3, 32, 32), sigma=torch.ones(1, 3), id_object=0)
    c = _config(4)
    c['object2']['wavelength'] = 6302.0
    dec = torchmfbd.Deconvolution(c)
    dec.add_frames(torch.rand(1, 3, 40, 40), sigma=torch.ones(1, 3), id_object=0)
    dec.add_frames(torch.rand(1, 3, 40, 40), sigma=torch.ones(1, 3), id_object=1)
    with pytest.raises(ValueError, match='one wavelength'):
        dec.deconvolve(infer_object=False, simultaneous_sequences=10, n_iterations=1)
