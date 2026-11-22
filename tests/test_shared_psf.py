"""Two objects seen through the same wavefronts (one wavelength, no diversity between them) get the PSFs of the first
object, computed once. Those PSFs must be bit for bit the ones computed for the second object on its own; an object at
another wavelength or with another diversity must keep its own."""
from pathlib import Path
import torch
import yaml
import torchmfbd

HERE = Path(__file__).parent


def _fit(make_synthetic_frames, monkeypatch, wavelength2=8542.0, diversity2=0.0):
    with open(HERE / "joint.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["optimization"]["gpu"] = -1
    cfg["object2"]["wavelength"] = wavelength2
    _, frames = make_synthetic_frames(cfg, 2, 5, 0.01, seed=4)
    orig = torchmfbd.Deconvolution.compute_psfs
    seen = []

    def spy(self, modes, diversity, jitter=None):
        psf, otf = orig(self, modes, diversity, jitter)
        flags = getattr(self, "_psf_same_as_first", None)
        if flags is not None:
            self._psf_same_as_first = None
            psf_alone, otf_alone = orig(self, modes, diversity, jitter)        # every object computed on its own
            self._psf_same_as_first = flags
            seen.append((list(flags), otf[1] is otf[0], torch.equal(otf[1], otf_alone[1]), torch.equal(psf[1], psf_alone[1])))
        return psf, otf

    monkeypatch.setattr(torchmfbd.Deconvolution, "compute_psfs", spy)
    d = torchmfbd.Deconvolution(cfg)
    d.add_frames(frames, id_object=0, id_diversity=0, diversity=0.0, sigma=0.01 * torch.ones(2, 5))
    d.add_frames(frames.flip(-1), id_object=1, id_diversity=0, diversity=diversity2, sigma=0.01 * torch.ones(2, 5))
    d.deconvolve(infer_object=False, optimizer="adam", simultaneous_sequences=10, n_iterations=2)
    assert seen, "compute_psfs never ran inside a fit: the check did not run"
    return seen


def test_same_wavelength_shares_identical_psfs(make_synthetic_frames, monkeypatch):
    for flags, shared, otf_same, psf_same in _fit(make_synthetic_frames, monkeypatch):
        assert flags == [False, True] and shared
        assert otf_same and psf_same, "the shared OTF of the second object is not the one computed for it on its own"


def test_other_wavelength_or_diversity_keeps_its_own(make_synthetic_frames, monkeypatch):
    for kw in (dict(wavelength2=6563.0), dict(diversity2=0.5)):
        for flags, shared, _, _ in _fit(make_synthetic_frames, monkeypatch, **kw):
            assert flags == [False, False] and not shared, f"{kw}: PSFs shared"
