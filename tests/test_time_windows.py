"""Does a step get the same window whatever the length of the series? The plan must depend only on the step's
neighbourhood and its absolute index, contain each step in its own window, reach a good step from a step below the
noise, and treat a series shorter than one window as a single window."""
import numpy as np
import pytest
from torchmfbd.time_windows import plan_time_windows, unique_windows, assemble_time_windows


def _snr(T, seed, weak_fraction=0.2):
    g = np.random.default_rng(seed)
    s = np.exp(g.normal(np.log(8.0), 1.0, T))
    s[g.random(T) < weak_fraction] = g.uniform(0.05, 0.9, int(np.sum(g.random(T) < weak_fraction)) or 1)[0]
    return s


@pytest.mark.parametrize("stride", [1, 2, 3, 4])
def test_each_step_is_in_its_window(stride):
    for seed in range(5):
        snr = _snr(49, seed)
        for t, (s, n) in enumerate(plan_time_windows(snr, stride=stride)):
            assert s <= t < s + n and 0 <= s and s + n <= 49


def test_sliding_windows_are_centred():
    plan = plan_time_windows(np.full(20, 50.0), base=5, stride=1)
    assert [s for s, _ in plan[2:18]] == [t - 2 for t in range(2, 18)]
    assert plan[0] == (0, 5) and plan[19] == (15, 5)


@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("cut", [(10, 20), (17, 15), (30, 7), (5, 3)])
def test_a_cut_series_keeps_the_windows(stride, cut):
    # The same steps planned as part of 49 or of a cut (with their absolute index): every step whose window and whose
    # search for a good step lie inside the cut gets the same window
    a, L = cut
    for seed in range(6):
        snr = _snr(49, seed)
        full = plan_time_windows(snr, stride=stride)
        part = plan_time_windows(snr[a:a + L], stride=stride, t0=a)
        for t in range(a, a + L):
            s, n = full[t]
            inside = a <= s and s + n <= a + L and (snr[t] >= 1.0 or (a <= t - 6 and t + 6 < a + L))
            if inside:
                assert (part[t - a][0] + a, part[t - a][1]) == (s, n), (seed, t, full[t], part[t - a])


def test_weak_step_reaches_a_good_one():
    snr = np.array([20.0] + [0.3] * 9 + [20.0])
    plan = plan_time_windows(snr, base=5, stride=1, reach=6)
    # step 4 is 4 from step 0 and 6 from step 10: its window runs from 0 to 4
    assert plan[4] == (0, 5)
    # step 6 is 4 from step 10: from 6 to 10
    assert plan[6] == (6, 5)
    # step 2 is 2 from step 0: the base window already holds step 0
    assert plan[2][0] == 0
    # nothing good within reach: the plain window
    far = plan_time_windows(np.array([20.0] + [0.3] * 15), base=5, stride=1, reach=6)
    assert far[10] == (8, 5)


def test_short_series_is_one_window():
    for T in (1, 2, 3, 4):
        plan = plan_time_windows(np.full(T, 5.0), base=5)
        assert plan == [(0, T)] * T


def test_stride_sets_the_number_of_fits():
    snr = np.full(49, 20.0)
    assert len(unique_windows(plan_time_windows(snr, stride=1))) == 45
    assert len(unique_windows(plan_time_windows(snr, stride=3))) in (16, 17)


def test_assemble_takes_each_step_from_its_window():
    plan = plan_time_windows(np.full(12, 20.0), base=5, stride=3)
    calls = []

    def fit(s, n):
        calls.append((s, n))
        return [(s, n, t) for t in range(s, s + n)]

    out = assemble_time_windows(plan, fit)
    assert sorted(calls) == unique_windows(plan)                  # each window fitted once
    assert [o[2] for o in out] == list(range(12))                 # each step is itself
    assert [o[:2] for o in out] == plan                           # ... from the window of the plan
    with pytest.raises(ValueError, match="returned"):
        assemble_time_windows(plan, lambda s, n: [0] * (n - 1))
