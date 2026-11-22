"""The speed-ups of 2026-11-22 must not change a single bit. The loss of the time path asks compute_object for the
unfiltered object only (need_filter=False), which skips the sums, mask and filtered object, and keeps the weighted frames
and SNR weights of a batch: its object must equal the one compute_object returns in full, in every evaluation. The
fixed-pattern removal runs its chunks of bursts on a thread pool: it must equal the chunks removed one by one."""
from pathlib import Path
import torch
import yaml
import torchmfbd
from torchmfbd import util

HERE = Path(__file__).parent
CUTOFF = 0.059 / (8542e-8 / 100.0 * 206265.0)


def _short_fit(make_synthetic_frames, monkeypatch, snr_threshold, check):
    """A 2-sequence, 4-step time fit (Adam, 3 iterations) with compute_object spied on: check(self, args, kwargs, out)
    is called on every call that asks for the unfiltered object only."""
    with open(HERE / "joint.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["optimization"]["gpu"] = -1
    cfg["optimization"]["filter_loss"] = "wiener_time"
    cfg.pop("object2", None)
    cfg["regularization"] = {"time": {"lambda": 0.05, "snr_threshold": snr_threshold}}
    n_t, n_frames = 4, 5
    _, frames = make_synthetic_frames(cfg, n_t, n_frames, 0.01, seed=3, static=True)
    sig = 0.01 * torch.ones(2, n_t, n_frames)
    sig[0, 1] = 0.05                                          # a noisy step, so that the SNR rule weights some links
    orig = torchmfbd.Deconvolution.compute_object
    calls = []

    def spy(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if kwargs.get("need_filter", True) is False:
            calls.append(1)
            check(self, args, kwargs, out, orig)
        return out

    monkeypatch.setattr(torchmfbd.Deconvolution, "compute_object", spy)
    d = torchmfbd.Deconvolution(cfg)
    d.add_frames(torch.stack([frames, frames.flip(-1)]), sigma=sig, id_object=0, id_diversity=0, diversity=0.0)
    d.deconvolve(infer_object=False, optimizer="adam", simultaneous_sequences=100, n_iterations=3)
    assert len(calls) >= 3, "the time path's loss did not ask for the unfiltered object: the check never ran"


def test_lean_time_object_is_the_full_one(make_synthetic_frames, monkeypatch):
    # In every evaluation of a fit, the object the loss gets (lean path, weighted frames kept for the batch) must be
    # bit for bit the unfiltered object of the full computation without the kept constants
    for thr in (12.0, 0.0):
        def check(self, args, kwargs, out, orig):
            kept = self._time_constants
            assert kept is not None and 0 in kept, "the weighted frames were not kept for the batch"
            self._time_constants = None
            full = orig(self, *args, **{**kwargs, "need_filter": True})[0]
            self._time_constants = kept
            for a, b in zip(out[0], full):
                assert torch.equal(a, b), f"threshold {thr}: lean object differs by {float((a - b).abs().max()):.3e}"
        _short_fit(make_synthetic_frames, monkeypatch, thr, check)


def test_kept_constants_belong_to_the_frames(make_synthetic_frames, monkeypatch):
    # Other frames passed in the same batch must not get the weighted frames kept for the first ones
    def check(self, args, kwargs, out, orig):
        images_ft, otf, sigma = args[0], args[1], args[2]
        other = [x * 1.5 for x in images_ft]
        with_kept = orig(self, other, otf, sigma, *args[3:], **kwargs)[0][0]
        kept = self._time_constants
        self._time_constants = None
        fresh = orig(self, other, otf, sigma, *args[3:], **kwargs)[0][0]
        self._time_constants = kept
        assert torch.equal(with_kept, fresh) and not torch.equal(with_kept, out[0][0])
    _short_fit(make_synthetic_frames, monkeypatch, 12.0, check)


def test_fixed_pattern_threads_equal_chunks():
    gen = torch.Generator().manual_seed(0)
    frames = 1.0 + 0.1 * torch.randn(20, 12, 48, 48, generator=gen)
    frames[..., ::5] += 0.02                                      # a column pattern for the removal to find
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        whole = util.remove_fixed_pattern(frames, CUTOFF, chunk=4)            # 5 chunks on the pool
        parts = [util.remove_fixed_pattern(frames[i:i + 4], CUTOFF, chunk=4) for i in range(0, 20, 4)]   # one chunk each
    finally:
        torch.set_num_threads(old)
    for k in range(3):
        assert torch.equal(whole[k], torch.cat([p[k] for p in parts])), f"output {k} differs"
    assert float(whole[2].sum()) > 0, "the test pattern was not found: the check would pass on an empty removal"
