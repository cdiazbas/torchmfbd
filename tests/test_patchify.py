import pytest
import warnings
import torchmfbd
import torch

def test_patchify():

    n_scans = 1
    n_frames = 2
    n_x = 80
    n_y = 80

    gen = torch.Generator().manual_seed(0)
    frames = torch.randn(n_scans, n_frames, n_x, n_y, generator=gen)

    p = torchmfbd.Patchify4D()

    for flatten in [True, False]:
        patches = p.patchify(frames, patch_size=20, stride_size=10, flatten_sequences=flatten)
        out = p.unpatchify(patches, apodization=0)

        assert frames.shape == out.shape, f'Expected {frames.shape}, got {out.shape}'
        assert torch.allclose(out, frames, atol=1e-6), f'Round trip differs by {torch.max(torch.abs(out - frames)).item()} (flatten_sequences={flatten})'

    # With apodization the mosaic is the region the trimmed patches cover, from pixel 3 to 77
    patches = p.patchify(frames, patch_size=20, stride_size=10, flatten_sequences=True)
    out = p.unpatchify(patches, apodization=3)

    assert out.shape == (n_scans, n_frames, 74, 74), f'Expected {(n_scans, n_frames, 74, 74)}, got {out.shape}'
    assert torch.allclose(out, frames[..., 3:77, 3:77], atol=1e-6), f'Round trip differs by {torch.max(torch.abs(out - frames[..., 3:77, 3:77])).item()}'

def test_patchify_non_divisible_stride():

    gen = torch.Generator().manual_seed(1)
    frames = torch.randn(1, 2, 82, 82, generator=gen)

    p = torchmfbd.Patchify4D()

    # (82 - 20) / 15 is not an integer: the last 2 rows and columns are not covered
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        patches = p.patchify(frames, patch_size=20, stride_size=15, flatten_sequences=True)
    assert len(w) > 0, 'No warning for a stride that does not cover the frame'

    out = p.unpatchify(patches, apodization=0)

    assert out.shape == (1, 2, 80, 80), f'Expected {(1, 2, 80, 80)}, got {out.shape}'
    assert torch.allclose(out, frames[..., 0:80, 0:80], atol=1e-6), f'Round trip differs by {torch.max(torch.abs(out - frames[..., 0:80, 0:80])).item()}'

def test_patchify_5d():

    gen = torch.Generator().manual_seed(2)
    frames = torch.randn(1, 2, 3, 40, 40, generator=gen)       # (s, t, f, x, y)

    p = torchmfbd.Patchify4D()

    for flatten in [True, False]:
        patches = p.patchify(frames, patch_size=20, stride_size=10, flatten_sequences=flatten)
        out = p.unpatchify(patches, apodization=0)

        assert frames.shape == out.shape, f'Expected {frames.shape}, got {out.shape}'
        assert torch.allclose(out, frames, atol=1e-6), f'Round trip differs by {torch.max(torch.abs(out - frames)).item()} (flatten_sequences={flatten})'

def test_patchify_cosine_weights():

    gen = torch.Generator().manual_seed(3)
    frames = torch.randn(1, 2, 60, 60, generator=gen)

    p = torchmfbd.Patchify4D()

    # stride = patch size: every pixel is covered by a single patch, so a zero weight would leave 0/0
    for stride in [20, 10]:
        patches = p.patchify(frames, patch_size=20, stride_size=stride, flatten_sequences=True)
        out = p.unpatchify(patches, apodization=0, weight_type='cosine', weight_params=5)

        assert torch.all(p.weight > 0), f'Cosine weight has {torch.sum(p.weight <= 0).item()} zero pixels'
        assert torch.allclose(out, frames, atol=1e-5), f'Round trip differs by {torch.max(torch.abs(out - frames)).item()} (stride {stride})'

def test_patchify_stride_too_large():

    frames = torch.zeros(1, 2, 80, 80)

    p = torchmfbd.Patchify4D()
    patches = p.patchify(frames, patch_size=20, stride_size=16, flatten_sequences=True)

    # The trimmed patches are 20 - 2*3 = 14 pixels wide; a stride of 16 leaves gaps
    try:
        p.unpatchify(patches, apodization=3)
    except ValueError:
        pass
    else:
        raise AssertionError('No ValueError for a stride larger than patch_size - 2*apodization')

def test_patchify_object():

    # One image per patch (a deconvolved object, (L, K, K)): the mosaic has no frame axis, as the examples plot it
    gen = torch.Generator().manual_seed(4)
    frames = torch.randn(1, 3, 80, 80, generator=gen)

    p = torchmfbd.Patchify4D()

    for flatten in [True, False]:
        patches = p.patchify(frames, patch_size=20, stride_size=10, flatten_sequences=flatten)
        obj = patches[..., 0, :, :]
        out = p.unpatchify(obj, apodization=0)

        assert out.shape == (1, 80, 80), f'Expected (1, 80, 80), got {out.shape} (flatten_sequences={flatten})'
        assert torch.allclose(out, frames[:, 0], atol=1e-6), f'Round trip differs by {torch.max(torch.abs(out - frames[:, 0])).item()} (flatten_sequences={flatten})'

    # With the frame axis kept, (L, 1, K, K), the frame axis is kept in the mosaic
    patches = p.patchify(frames, patch_size=20, stride_size=10, flatten_sequences=True)
    out = p.unpatchify(patches[:, 0:1], apodization=0)
    assert out.shape == (1, 1, 80, 80), f'Expected (1, 1, 80, 80), got {out.shape}'
