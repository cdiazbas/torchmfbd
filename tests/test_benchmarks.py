"""The benchmark's metrics and scenes (no fits: seconds)."""
import pathlib, sys
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))
import metrics as M
import scenes as S


@pytest.fixture
def truth():
    rng = np.random.default_rng(0)
    return M.truth_units(1 + 0.1 * rng.standard_normal((128, 128)))


def test_perfect_object_scores_zero(truth):
    assert M.object_error(truth, truth) < 1e-12
    assert max(M.band_error(truth, truth)) < 1e-12


def test_contrast_error(truth):
    o = 1 + 1.3 * (truth - 1)
    assert M.object_error(o, truth) == pytest.approx(0.3, abs=1e-6)
    assert M.object_error_bestscale(o, truth) < 1e-6


@pytest.mark.parametrize("scale", [0.8, 0.95, 1.05])
def test_units_mismatch_raises(truth, scale):
    with pytest.raises(ValueError):
        M.object_error(truth, scale * truth)


def test_units_guard_threshold(truth):
    M.object_error(1.02 * truth, truth)                    # inside RATIO_TOL (3 %): scored
    with pytest.raises(ValueError):
        M.object_error(1.04 * truth, truth)


def test_folded_psf_reference():
    rng = np.random.default_rng(1)
    p = rng.random((128, 128)); p /= p.sum()
    w = M.fold_psf(p, 64)
    assert w.sum() == pytest.approx(1.0)
    # the folded PSF's OTF is the full OTF sampled on the 64-px grid
    assert np.allclose(np.fft.fft2(w), np.fft.fft2(p)[::2, ::2])
    assert M.psf_error(w[None, None], p[None]) < 1e-12


def test_transfer_error():
    rng = np.random.default_rng(2)
    p = rng.random((3, 64, 64)); p /= p.sum(axis=(-1, -2), keepdims=True)
    t = rng.random((2, 64, 64))
    assert M.transfer_error(np.stack([p, p]), np.fft.fft2(p), t) < 1e-10
    q = np.roll(p, 1, axis=-1)                               # every frame off by one pixel
    assert M.transfer_error(np.stack([q, q]), np.fft.fft2(p), t) > 0.5


def test_control():
    M.check_control(0.05, 0.10)
    with pytest.raises(RuntimeError):
        M.check_control(0.095, 0.10)


def test_scenes_reproducible():
    a, b = S.synthetic(3, n_frames=2), S.synthetic(3, n_frames=2)
    assert np.array_equal(a["frames"], b["frames"]) and a["shifts"] == b["shifts"]
    assert not np.array_equal(a["frames"], S.synthetic(4, n_frames=2)["frames"])


def test_scene_truth_in_frame_units():
    sc = S.synthetic(0, n_frames=2)
    assert sc["truth"].mean() == pytest.approx(sc["frames"].mean() / sc["frames"].mean(), rel=1e-3)
    assert sc["psfs"].sum(axis=(-1, -2)) == pytest.approx(np.ones(2))
