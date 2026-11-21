"""Windows of consecutive time steps for fitting a long series a piece at a time.

A series too long for one batch on the GPU is fitted in windows of a few steps, the time term tying the steps of a
window. How the windows are laid out decides whether a step comes out the same when the series is cut differently:
windows that start at the first step of each series (7 steps every 4) changed the same step by up to 31 % between a
series of 49 steps and the same steps cut to 15 (2026-11-21). Here each step is taken from one window placed by the
absolute index of the step, so a step whose window fits in both series is fitted from the same frames in both.
"""
import numpy as np

__all__ = ['plan_time_windows', 'unique_windows', 'assemble_time_windows']


def plan_time_windows(snr, base=5, stride=3, reach=6, t0=0, good=12.0, weak=1.0):
    """
    The window each time step of a series is taken from.

    Windows of `base` steps are centred every `stride` steps on the absolute step index (t0 + t), and each step is
    taken from the window whose centre is nearest, at most stride // 2 steps away. A step whose frames are below the
    noise (SNR < weak) gets instead a window that reaches the nearest step with SNR >= good, if one lies within `reach`
    steps: such a step has almost nothing of its own and leans on that step through the time term. A series shorter
    than `base` is one window.

    Measured on 49-step series with a known truth (2026-11-21): base 5 is best in realistic seeing (7 and 9 worse by 6
    and 13 %); stride 3 costs 0.7 % against one window per step with 2.3 times fewer fits; reach 6 is 1/(2f) with
    f ~ 0.08 the fraction of power the Sun changes per step, and recovers bad steps up to about 6 steps from a good one.

    Parameters
    ----------
    snr : array-like
        (n_t,) signal-to-noise ratio of each step (torchmfbd.step_snr of its frames).
    base, stride, reach : int
        Steps per window, steps between window centres, largest distance to a good step.
    t0 : int
        Absolute index of the first step: keep it when cutting a series so that the windows stay where they were.
    good, weak : float
        SNR of a step another can lean on, and SNR below which a step leans on one.

    Returns
    -------
    list of (int, int)
        (first step, number of steps) of the window of each step, in indices of this series.
    """
    snr = np.asarray(snr, dtype=float)
    T = len(snr)
    if T == 0:
        return []
    if base < 1 or stride < 1 or reach < 0:
        raise ValueError(f"plan_time_windows needs base >= 1, stride >= 1 and reach >= 0, got {base}, {stride}, {reach}")
    wb = min(base, T)
    good_steps = np.where(snr >= good)[0]
    out = []
    for t in range(T):
        c = t - ((t + t0) % stride) + stride // 2 if stride > 1 else t
        s, n = min(max(c - wb // 2, 0), T - wb), wb
        if snr[t] < weak:
            near = [g for g in good_steps if abs(g - t) <= reach]
            if near:
                g = min(near, key=lambda g: abs(g - t))
                a, b = min(t, g), max(t, g)
                if b - a + 1 < wb:
                    a = max(0, min(a, b - wb + 1))
                    b = a + wb - 1
                s, n = a, b - a + 1
        out.append((int(s), int(n)))
    return out


def unique_windows(plan):
    """The distinct windows of a plan, in order of their first step: the fits a series needs."""
    return sorted(set(plan))


def assemble_time_windows(plan, fit_window, map=map):
    """
    Fit every distinct window of a plan once and take each step from its own window.

    The windows are independent fits, so they can run at the same time: pass as `map` the map of a pool whose
    workers each own a GPU (for instance concurrent.futures.ProcessPoolExecutor(...).map with an initializer that
    sets the device). Each window must always land on the same type of GPU for the result to be reproducible: the
    same fit differs by up to 9 % between a T4 and a V100S (2026-11-21).

    Parameters
    ----------
    plan : list of (int, int)
        From plan_time_windows.
    fit_window : callable
        fit_window(first, n) fits steps first ... first + n - 1 together and returns a sequence of n per-step results
        (objects, or tuples of what the caller needs).
    map : callable
        map(function, starts, lengths) used to run the fits (two iterables, as map and Executor.map take them); the
        built-in map runs them one after another. With a process pool, fit_window must be a module-level function.

    Returns
    -------
    list
        The result of each step of the series, from the window the plan gives it.
    """
    windows = unique_windows(plan)
    results = list(map(fit_window, [s for s, _ in windows], [n for _, n in windows]))     # picklable for process pools
    fits = {}
    for (s, n), res in zip(windows, results):
        if len(res) != n:
            raise ValueError(f"fit_window({s}, {n}) returned {len(res)} results for {n} steps")
        fits[(s, n)] = res
    return [fits[(s, n)][t - s] for t, (s, n) in enumerate(plan)]
