"""Does the SNR rule of the time weight do what it says, and nothing when it is off? A link (t, t+1) gets the time
weight times threshold_v / min(SNR_t, SNR_t+1), at least 1 and at most 100, with threshold_v = min(threshold, best SNR
of the series) for snr_reference 'window'. Off (no snr_threshold, or 0), the fit must be the same bit for bit. Whether
the rule improves the objects is measured on scenes with a truth (exchange/, 2026-11-20/21), not here."""
import pytest
import torch
import torchmfbd
from torchmfbd.reg_time import snr_link_factors, SNR_RULE_CAP
from test_time_solver import _random_system, _dense, _series_fit, dec  # noqa: F401  (dec is a fixture)


def test_link_factors():
    snr = torch.tensor([[20.0, 0.5, 20.0], [2.0, 1.0, 2.0], [20.0, 1e-4, 30.0]], dtype=torch.float64)
    w = snr_link_factors(snr, 12.0, "window")
    f = snr_link_factors(snr, 12.0, "fixed")
    # a weak step next to good ones: 12 / 0.5 on both of its links, with either reference
    torch.testing.assert_close(w[0], torch.tensor([24.0, 24.0], dtype=torch.float64))
    torch.testing.assert_close(f[0], torch.tensor([24.0, 24.0], dtype=torch.float64))
    # every step noisy: relative to the best of the series (2 / 1), not to 12
    torch.testing.assert_close(w[1], torch.tensor([2.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(f[1], torch.tensor([12.0, 12.0], dtype=torch.float64))
    # a step without data is capped
    assert torch.all(w[2] == SNR_RULE_CAP)
    # good steps keep the plain weight
    assert torch.all(snr_link_factors(torch.full((1, 4), 50.0), 12.0) == 1.0)
    with pytest.raises(ValueError, match="window' or 'fixed"):
        snr_link_factors(snr, 12.0, "best")


@pytest.mark.parametrize("M", [2, 3, 6])
def test_per_link_solver_matches_dense(dec, M):
    S, D, reg = _random_system(3, M, 4, 6, seed=50 + M)
    lam = torch.rand(3, M - 1, generator=torch.Generator().manual_seed(M), dtype=torch.float64) * 5.0 + 0.1
    out = dec.solve_object_time_evolution_batched(S, D, reg, 0.7, lam)
    for s in range(3):
        torch.testing.assert_close(out[s], _dense(S[s], D[s], reg, 0.7, lam[s].tolist()), rtol=1e-9, atol=1e-12)


def test_equal_links_match_the_scalar_weight(dec):
    S, D, reg = _random_system(2, 5, 3, 6, seed=60)
    scalar = dec.solve_object_time_evolution_batched(S, D, reg, 0.7, 2.5)
    links = dec.solve_object_time_evolution_batched(S, D, reg, 0.7, torch.full((2, 4), 2.5, dtype=torch.float64))
    torch.testing.assert_close(links, scalar, rtol=1e-14, atol=1e-15)


def test_off_is_bitwise_unchanged(make_synthetic_frames):
    e_absent, _ = _series_fit(make_synthetic_frames, 5e-4, bad_factor=4.0, sigma=0.02, iterations=10)
    e_zero, _ = _series_fit(make_synthetic_frames, 5e-4, bad_factor=4.0, sigma=0.02, iterations=10, time_extra={"snr_threshold": 0})
    assert torch.equal(e_absent, e_zero)


def test_rule_weights_the_links_of_the_noisy_step(make_synthetic_frames, monkeypatch):
    # Steps of SNR ~70 around one of SNR ~6 (noise 0.02, the middle step 4 times noisier): with threshold 12 the two
    # links of the noisy step get about 12 / 6 = 2 times the weight, the others exactly the plain weight
    seen = []
    solve = torchmfbd.Deconvolution.solve_object_time_evolution_batched

    def spy(self, S, D, Reg_Space, lambda_s, lambda_t):
        seen.append(lambda_t)
        return solve(self, S, D, Reg_Space, lambda_s, lambda_t)

    monkeypatch.setattr(torchmfbd.Deconvolution, "solve_object_time_evolution_batched", spy)
    _series_fit(make_synthetic_frames, 5e-4, bad_factor=4.0, sigma=0.02, iterations=2, time_extra={"snr_threshold": 12})
    lam = seen[-1]
    assert torch.is_tensor(lam) and lam.shape == (1, 4)
    base = 5e-4 * (64.0 / 32)**2 * (5 / 12.0)          # the configured weight scaled to 32-px patches and 5 frames
    ratio = (lam[0] / base).tolist()
    assert ratio[0] == pytest.approx(1.0) and ratio[3] == pytest.approx(1.0), ratio
    assert 1.5 < ratio[1] < 3.0 and 1.5 < ratio[2] < 3.0, ratio


@pytest.mark.parametrize("extra,msg", [({"snr_threshold": -1}, ">= 0"), ({"snr_threshold": 12, "snr_reference": "best"}, "window' or 'fixed")])
def test_config_is_checked(extra, msg):
    import yaml
    from test_time_solver import HERE
    with open(HERE / "joint.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["optimization"]["gpu"] = -1
    cfg["optimization"]["filter_loss"] = "wiener_time"
    cfg["regularization"] = {"time": {"lambda": 5e-4, **extra}}
    d = torchmfbd.Deconvolution(cfg)
    with pytest.raises(ValueError, match=msg):
        d.set_regularizations()         # the regularization block is read when the fit starts


def test_contrast_filter_refuses_the_rule():
    import yaml
    from test_time_solver import HERE
    with open(HERE / "joint.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["optimization"]["gpu"] = -1
    cfg["optimization"]["filter_loss"] = "wiener_time_contrast"
    cfg["regularization"] = {"time": {"lambda": 5e-4, "snr_threshold": 12}}
    d = torchmfbd.Deconvolution(cfg)
    with pytest.raises(ValueError, match="needs filter_loss wiener_time"):
        d.set_regularizations()
