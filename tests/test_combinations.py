"""Do the options work together? Every combination of object filter x loss x patch coupling x wavefront restriction x
optimizer either fits (finite loss that decreases, finite object) or refuses with a clear error. This checks that the
options run together, not that they help: that is measured with benchmarks/run.py."""
from pathlib import Path
import itertools
import pytest
import torch
import yaml
import torchmfbd

HERE = Path(__file__).parent
PAIRS = [(0, 1), (1, 2), (2, 3)]
NULLSPACE_UNSUPPORTED = ("marginal",)          # the restriction is written for the joint loss (docs: config.rst)


def _config(loss, filt, coupling, nullspace):
    with open(HERE / f"{loss}.yaml") as f:
        c = yaml.safe_load(f)
    c["optimization"]["gpu"] = -1
    c.pop("object2", None)
    c["object1"]["image_filter"] = filt
    c["regularization"] = {"coupling": {"lambda": 30.0}} if coupling else {}
    if nullspace:
        c["psf"]["nullspace_d_r0"] = 5.0
    return c


@pytest.mark.parametrize("loss,filt,coupling,nullspace,optimizer",
                         list(itertools.product(["joint", "marginal"], ["scharmer", "scharmer_binary", "garrote", "tophat"],
                                                [False, True], [False, True], ["lbfgs", "adam"])))
def test_options_together(make_synthetic_frames, loss, filt, coupling, nullspace, optimizer):
    cfg = _config(loss, filt, coupling, nullspace)
    obj, frames = make_synthetic_frames(_config(loss, "scharmer", False, False), 4, 5, 0.005, seed=0)
    torch.manual_seed(0)
    d = torchmfbd.Deconvolution(cfg)
    d.add_frames(frames, sigma=0.005 * torch.ones(4, 5), id_object=0, id_diversity=0, diversity=0.0)
    if coupling:
        d.set_neighbours(PAIRS)
    n_it = 3 if optimizer == "lbfgs" else 20
    if nullspace and loss in NULLSPACE_UNSUPPORTED:
        with pytest.raises(ValueError):
            d.deconvolve(infer_object=False, optimizer=optimizer, simultaneous_sequences=200, n_iterations=n_it)
        return
    d.deconvolve(infer_object=False, optimizer=optimizer, simultaneous_sequences=200, n_iterations=n_it)
    assert torch.all(torch.isfinite(d.loss)), "non-finite loss"
    assert d.loss[-1] < d.loss[0], f"loss did not decrease: {d.loss[0].item()} -> {d.loss[-1].item()}"
    assert torch.all(torch.isfinite(d.obj[0])), "non-finite object"


def _time_fit(make_synthetic_frames, filt, coupling, lam_t=1.0, lam_c=30.0):
    cfg = _config("joint", filt, coupling, False)
    if coupling:
        cfg["regularization"]["coupling"]["lambda"] = lam_c
    cfg["optimization"]["filter_loss"] = "wiener_time"
    cfg["regularization"]["time"] = {"lambda": lam_t}
    obj, frames = make_synthetic_frames(_config("joint", "scharmer", False, False), 4, 5, 0.005, seed=0)
    d = torchmfbd.Deconvolution(cfg)
    # four sequences as 2 patches x 2 time steps, (s, t, f, x, y)
    d.add_frames(frames.reshape(2, 2, 5, 32, 32), sigma=0.005 * torch.ones(2, 2, 5), id_object=0, id_diversity=0, diversity=0.0)
    if coupling:
        d.set_neighbours([(0, 1)])
    d.deconvolve(infer_object=False, optimizer="adam", simultaneous_sequences=200, n_iterations=20)
    return d


@pytest.mark.parametrize("filt", ["scharmer", "scharmer_binary", "garrote", "tophat"])
def test_time_with_each_filter(make_synthetic_frames, filt):
    d = _time_fit(make_synthetic_frames, filt, False)
    assert torch.all(torch.isfinite(d.loss)) and d.loss[-1] < d.loss[0]
    assert torch.all(torch.isfinite(d.obj[0]))


@pytest.mark.parametrize("filt", ["scharmer", "scharmer_binary", "garrote", "tophat"])
def test_patch_coupling_with_time(make_synthetic_frames, filt):
    d = _time_fit(make_synthetic_frames, filt, True)
    assert torch.all(torch.isfinite(d.loss)) and d.loss[-1] < d.loss[0]
    assert torch.all(torch.isfinite(d.obj[0]))


def test_patch_coupling_with_time_pairs_within_each_step(make_synthetic_frames):
    # 2 patches x 2 steps, flattened (s t): patch 0 is sequences 0 (t0) and 1 (t1), patch 1 is 2 and 3. A strong
    # coupling of patches 0 and 1 must pull 0 towards 2 and 1 towards 3, not 0 towards 1 (different steps)
    def spread(d, a, b):
        m = d.modes_seq[0][..., 2:10]
        return float(torch.sum((m[a] - m[b])**2))
    free = _time_fit(make_synthetic_frames, "scharmer", False)
    tied = _time_fit(make_synthetic_frames, "scharmer", True, lam_c=1e6)
    assert spread(tied, 0, 2) < 0.01 * spread(free, 0, 2)
    assert spread(tied, 1, 3) < 0.01 * spread(free, 1, 3)
    # a strong coupling also shrinks the modes overall, so compare within the tied fit: the pairs of one step must end
    # far closer than the same patch at two steps
    assert spread(tied, 0, 2) < 0.1 * spread(tied, 0, 1)
    assert spread(tied, 1, 3) < 0.1 * spread(tied, 2, 3)
