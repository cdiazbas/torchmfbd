import numpy as np
import torch
import torchmfbd


def _frames(n_f=3, n=64, seed=0):
    rng = np.random.default_rng(seed)
    base = torch.tensor(rng.standard_normal((n, n)), dtype=torch.float32)
    base = torch.nn.functional.avg_pool2d(base[None, None], 5, 1, 2)[0, 0]
    shifts = [(0, 0), (1, 0), (0, -1)][:n_f]
    return torch.stack([torch.roll(base, s, (0, 1)) for s in shifts])[None, None]       # (1, 1, n_f, n, n)


def test_apply_destretch_takes_the_flow_destretch_returns():
    fr = _frames()
    warped, tt = torchmfbd.destretch(fr, ngrid=8, lr=0.1, reference_frame=0, border=6, n_iterations=5, lambda_tt=0.01)
    assert tt.dim() == 5                                    # (n_seq, n_f, 2, n_x, n_y)
    again = torchmfbd.apply_destretch(fr, tt)               # the second object of a burst gets the flow of the first
    assert again.shape == fr.shape
    flat = torchmfbd.apply_destretch(fr, tt.reshape(-1, *tt.shape[2:]))
    assert torch.allclose(again, flat)
    assert torch.allclose(again, warped, atol=1e-4)         # the same flow on the same frames gives destretch's output
