import numpy as np
import torch
import torchmfbd


def _moving_frames(offsets, n=96, big=128, seed=0):
    rng = np.random.default_rng(seed)
    scene = rng.standard_normal((big, big)).astype('float32')
    c = (big - n) // 2
    return torch.tensor(np.stack([scene[c + oy:c + oy + n, c + ox:c + ox + n] for oy, ox in offsets]))[None]


def test_follow_motion_undoes_a_known_shift():
    """One frame of five moved by (3, -2) px: every patch of that frame is cut again where it moved, so it equals the
    patch of the frames that did not move, and the shift is recorded."""
    offsets = [(0, 0), (0, 0), (3, -2), (0, 0), (0, 0)]
    x = _moving_frames(offsets)
    P = torchmfbd.Patchify4D()
    p = P.patchify(x, patch_size=32, stride_size=16, follow_motion=8)
    assert torch.all(P.motion_shifts[0, :, 2] == torch.tensor([-3, 2])), P.motion_shifts[0, :, 2]
    assert torch.all(P.motion_shifts[0, :, [0, 1, 3, 4]] == 0)
    interior = [k for k in range(p.shape[0]) if 8 <= (k // P.Ly) * 16 <= 96 - 32 - 8 and 8 <= (k % P.Ly) * 16 <= 96 - 32 - 8]
    assert len(interior) > 0
    for k in interior:
        assert torch.equal(p[k, 2], p[k, 0]), f'patch {k}'


def test_follow_motion_off_is_unchanged():
    x = _moving_frames([(0, 0), (1, 2), (-2, 1)])
    a = torchmfbd.Patchify4D().patchify(x, patch_size=32, stride_size=16)
    b = torchmfbd.Patchify4D().patchify(x, patch_size=32, stride_size=16, follow_motion=0)
    assert torch.equal(a, b)


def test_follow_motion_shared_shifts():
    """Two objects that share the wavefront (two cameras seeing the same motion with different noise) are cut with the
    shifts estimated on the first: the second gets exactly those shifts, and its moved frame is cut back onto the others."""
    offsets = [(0, 0), (0, 0), (3, -2), (0, 0), (0, 0)]
    a = _moving_frames(offsets, seed=0)
    b = a + 0.3 * torch.randn(a.shape, generator=torch.Generator().manual_seed(5))
    Pa, Pb = torchmfbd.Patchify4D(), torchmfbd.Patchify4D()
    Pa.patchify(a, patch_size=32, stride_size=16, follow_motion=8)
    pb = Pb.patchify(b, patch_size=32, stride_size=16, motion_shifts=Pa.motion_shifts)
    assert torch.equal(Pb.motion_shifts, Pa.motion_shifts)
    # frame 2 of b, cut by hand at each grid position plus the recorded shift (-3, 2), where no padding is involved
    for k in range(pb.shape[0]):
        y0, x0 = (k // Pb.Ly) * 16, (k % Pb.Ly) * 16
        if 3 <= y0 and y0 - 3 + 32 <= 96 and 0 <= x0 + 2 and x0 + 2 + 32 <= 96:
            assert torch.equal(pb[k, 2], b[0, 2, y0 - 3:y0 - 3 + 32, x0 + 2:x0 + 2 + 32]), f"patch {k}"
