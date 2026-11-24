"""Does the time path solve what it claims to solve? The object of every time step minimises, per frequency,
sum_t sum_j |D_tj - S_tj O_t|^2 + lambda_s R |O_t|^2 + sum_t lambda_(t,t+1) |O_t+1 - O_t|^2, a tridiagonal system in t.
These tests check the two solvers against that system written out densely and solved with torch.linalg.solve, the
two solvers against each other, and the fit against a known object: the algebra and the physics, not stored numbers,
so they keep their meaning when the code around them changes."""
from pathlib import Path
import pytest
import torch
import yaml
import torchmfbd

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def dec():
    with open(HERE / "joint.yaml") as f:
        config = yaml.safe_load(f)
    config["optimization"]["gpu"] = -1
    config["optimization"]["filter_loss"] = "wiener_time"
    config["regularization"] = {"time": {"lambda": 1.0}}
    return torchmfbd.Deconvolution(config)


def _random_system(n_s, M, J, N, seed):
    """Random OTFs and data in float64 (every frequency carries data) and a positive spatial weight."""
    g = torch.Generator().manual_seed(seed)
    c = lambda *s: torch.complex(torch.randn(*s, generator=g, dtype=torch.float64), torch.randn(*s, generator=g, dtype=torch.float64))
    S, D = c(n_s, M, J, N, N), c(n_s, M, J, N, N)
    reg = torch.rand(N, N, generator=g, dtype=torch.float64) + 0.1
    return S, D, reg


def _dense(S, D, reg, lambda_s, lam_pairs):
    """The tridiagonal system of one sequence built as a dense M x M matrix per frequency and solved directly.
    S, D: (M, J, N, N); lam_pairs: (M - 1,) weights of the links (t, t+1)."""
    M, J, N, _ = S.shape
    A = torch.sum(torch.abs(S)**2, dim=1).reshape(M, -1)
    b = torch.sum(torch.conj(S) * D, dim=1).reshape(M, -1)
    L = torch.zeros(M, M, dtype=torch.float64)
    for t, lam in enumerate(lam_pairs):
        L[t, t] += lam; L[t + 1, t + 1] += lam; L[t, t + 1] -= lam; L[t + 1, t] -= lam
    out = torch.empty_like(b)
    for k in range(b.shape[1]):
        Mk = torch.diag(A[:, k] + lambda_s * reg.flatten()[k]) + L
        out[:, k] = torch.linalg.solve(Mk.to(torch.complex128), b[:, k])
    return out.reshape(M, N, N)


@pytest.mark.parametrize("M", [1, 2, 3, 7])
def test_batched_solver_matches_dense(dec, M):
    S, D, reg = _random_system(3, M, 4, 6, seed=M)
    lambda_s, lambda_t = 0.7, 2.5
    out = dec.solve_object_time_evolution_batched(S, D, reg, lambda_s, lambda_t)
    for s in range(S.shape[0]):
        ref = _dense(S[s], D[s], reg, lambda_s, [lambda_t] * (M - 1))
        torch.testing.assert_close(out[s], ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("M", [1, 2, 3, 7])
def test_single_solver_matches_dense(dec, M):
    S, D, reg = _random_system(1, M, 4, 6, seed=10 + M)
    lambda_s, lambda_t = 0.7, 2.5
    out = dec.solve_object_time_evolution(S[0], D[0], reg, lambda_s, lambda_t)
    torch.testing.assert_close(out, _dense(S[0], D[0], reg, lambda_s, [lambda_t] * (M - 1)), rtol=1e-9, atol=1e-12)


def test_contrast_weighted_solver_matches_dense(dec):
    # wiener_time_contrast: the link (t, t+1) gets lambda / (c_t c_t+1) with the contrasts normalised to their minimum
    M = 5
    S, D, reg = _random_system(1, M, 4, 6, seed=20)
    contrast = torch.tensor([1.0, 0.5, 2.0, 1.5, 0.8], dtype=torch.float64)
    cn = contrast / contrast.min()
    lam = (2.5 / (cn[:-1] * cn[1:])).tolist()
    out = dec.solve_object_time_evolution(S[0], D[0], reg, 0.7, 2.5, contrast=contrast)
    torch.testing.assert_close(out, _dense(S[0], D[0], reg, 0.7, lam), rtol=1e-9, atol=1e-12)


def test_solvers_agree(dec):
    # The batched solver is the single-sequence one with the sequences as a leading dimension
    S, D, reg = _random_system(4, 6, 3, 8, seed=30)
    out = dec.solve_object_time_evolution_batched(S, D, reg, 0.3, 1.7)
    for s in range(4):
        torch.testing.assert_close(out[s], dec.solve_object_time_evolution(S[s], D[s], reg, 0.3, 1.7), rtol=1e-12, atol=1e-14)


def test_frequencies_without_data_are_zero(dec):
    # Beyond the cutoff sum|S|^2 = 0 in every step: the system is singular there and the object must be 0, not NaN
    S, D, reg = _random_system(2, 4, 3, 6, seed=40)
    S[..., 0, 1] = 0.0
    out = dec.solve_object_time_evolution_batched(S, D, torch.zeros_like(reg), 0.0, 1.0)
    assert torch.all(torch.isfinite(out))
    assert torch.all(out[..., 0, 1] == 0)


def _series_fit(make_synthetic_frames, lam_t, bad_factor=10.0, n_t=5, n_frames=5, sigma=0.005, seed=0, iterations=30, time_extra=None):
    """One patch, n_t time steps of the same object, each through its own seeing; the middle step has bad_factor
    times the noise. time_extra: more keys of the `time` regularization. Returns the rms error of each step's object
    against the truth (diffraction mask, no border) and the index of the noisy step."""
    with open(HERE / "joint.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["optimization"]["gpu"] = -1
    cfg.pop("object2", None)
    obj, frames = make_synthetic_frames(cfg, n_t, n_frames, sigma, seed=seed, static=True)
    g = torch.Generator().manual_seed(seed + 100)
    bad = n_t // 2
    extra = sigma * (bad_factor**2 - 1)**0.5
    frames[bad] = frames[bad] + extra * torch.randn(frames[bad].shape, generator=g)
    sig = sigma * torch.ones(1, n_t, n_frames)
    sig[0, bad] = sigma * bad_factor
    cfg["optimization"]["filter_loss"] = "wiener_time"
    cfg["regularization"] = {"time": {"lambda": lam_t, **(time_extra or {})}}
    d = torchmfbd.Deconvolution(cfg)
    d.add_frames(frames[None], sigma=sig, id_object=0, id_diversity=0, diversity=0.0)
    d.deconvolve(infer_object=False, optimizer="adam", simultaneous_sequences=200, n_iterations=iterations)
    mask = d.mask_diffraction_th[0].cpu()
    truth = torch.fft.ifft2(torch.fft.fft2(obj[0]) * mask).real
    b = d.npix_apod
    err = torch.sqrt(torch.mean((d.obj[0][0].cpu() - truth)[:, b:-b, b:-b]**2, dim=(-1, -2)))
    return err, bad


def test_time_term_rescues_bad_step(make_synthetic_frames):
    # With a Sun that does not change, the time term must bring the noisy step towards its neighbours: its error falls
    # well below the fit without the time term (to 0.06-0.07 of it on 2026-11-21, seeds 0-2), and the good steps do not
    # get worse
    e0, bad = _series_fit(make_synthetic_frames, 0.0)
    e1, _ = _series_fit(make_synthetic_frames, 0.05)
    good = [t for t in range(len(e0)) if t != bad]
    assert e1[bad] < 0.5 * e0[bad], f"bad step: {e0[bad]:.4f} without the time term, {e1[bad]:.4f} with it"
    assert torch.all(e1[good] < 1.1 * e0[good]), f"good steps: {e0[good].tolist()} -> {e1[good].tolist()}"


def test_time_fit_accuracy_bar(make_synthetic_frames):
    # Absolute quality of a short time-series fit against the truth. The bar is the error measured on 2026-11-21
    # (CPU, torch 2.7.0; seeds 1 and 2 give 0.0233 and 0.0207) times 1.2: it catches a large degradation, not the drift between torch versions
    err, _ = _series_fit(make_synthetic_frames, 0.05)
    assert err.mean() < 0.0232 * 1.2, f"mean object error {err.mean():.5f}, bar {0.0232} x 1.2"


def _ramped_series(make_synthetic_frames, lam_t, link_plane, monkeypatch, n_t=4, n_frames=5, sigma=0.005, seed=0):
    """A Sun that does not change, seen through a gradient that does (transparency, a limb, scattered light): each step
    multiplies its frames by its own ramp. remove_gradient_apodization on; link_plane False takes the plane out of the
    time link, as before. Returns the restored objects (n_t, n, n) without the apodization border."""
    with open(HERE / "joint.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["optimization"]["gpu"] = -1
    cfg["images"]["remove_gradient_apodization"] = True
    cfg.pop("object2", None)
    obj, frames = make_synthetic_frames(cfg, n_t, n_frames, sigma, seed=seed, static=True)
    n = frames.shape[-1]
    x = torch.linspace(-0.5, 0.5, n)
    for t in range(n_t):
        frames[t] = frames[t] * (1.0 + 0.2 * (t - (n_t - 1) / 2) / n_t * x[None, None, :])
    cfg["optimization"]["filter_loss"] = "wiener_time"
    cfg["regularization"] = {"time": {"lambda": lam_t, "snr_threshold": 0}}
    with monkeypatch.context() as m:
        if not link_plane:
            m.setattr(torchmfbd.Deconvolution, "plane_spectrum", lambda self, plane, i: None)
        torch.manual_seed(0)
        d = torchmfbd.Deconvolution(cfg)
        d.add_frames(frames[None], sigma=sigma * torch.ones(1, n_t, n_frames), id_object=0, id_diversity=0, diversity=0.0)
        d.deconvolve(infer_object=False, optimizer="adam", simultaneous_sequences=200, n_iterations=10)
    b = d.npix_apod
    return d.obj[0][0].cpu()[:, b:-b, b:-b]


def test_time_link_without_weight_keeps_the_objects(make_synthetic_frames, monkeypatch):
    # O_t = Q_t - P_t with the data D + S P is the same object as before when nothing links the steps
    a = _ramped_series(make_synthetic_frames, 0.0, True, monkeypatch)
    b = _ramped_series(make_synthetic_frames, 0.0, False, monkeypatch)
    assert torch.allclose(a, b, rtol=0, atol=1e-4 * b.abs().max()), f"largest difference {(a - b).abs().max():.2e}"


def test_time_link_includes_the_plane(make_synthetic_frames, monkeypatch):
    # With a strong link the whole objects, plane included, come together: the steps of an unchanging Sun seen through
    # changing gradients differ much less than when the link sees the objects without their planes, which keep each
    # step's gradient
    a = _ramped_series(make_synthetic_frames, 50.0, True, monkeypatch)
    b = _ramped_series(make_synthetic_frames, 50.0, False, monkeypatch)
    spread = lambda o: float(torch.sqrt(torch.mean((o - o.mean(dim=0))**2)))
    assert spread(a) < 0.3 * spread(b), f"spread between steps {spread(a):.2e} with the plane in the link, {spread(b):.2e} without"
