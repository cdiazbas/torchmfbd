import torch
from torchmfbd.regularization import Regularization

class RegularizationTime(Regularization):

    def __init__(self, lambda_reg=0.0, variable=None):
        super().__init__('time', variable, lambda_reg)
        
        self.lambda_reg = lambda_reg

    def __call__(self, x):
        # The time coupling is solved inside compute_object (filter_loss wiener_time*); this class is only a
        # placeholder and must not be added as a loss term that silently contributes zero
        raise NotImplementedError("RegularizationTime has no loss term: use filter_loss wiener_time with a `time` regularization entry")

# Threshold of the SNR rule when the time regularization does not give one: 1/f, with f ~ 0.08 the fraction of power the
# Sun changes per step in the CRISP 8542 sequence. On by default since the validation of 2026-11-21: better than one
# weight in 12 of 12 new 49-step series (mean -23.7 %) and the same as the reference restoration on the real sequence.
# snr_threshold: 0 in the configuration gives one weight for every link, as before.
SNR_RULE_DEFAULT_THRESHOLD = 12.0

# Largest factor by which the SNR rule raises the time weight of a link. Not tuned: it only caps steps with no data at
# all, and on 2026-11-20/21 no measured scene reached it with the window reference
SNR_RULE_CAP = 100.0


def snr_link_factors(snr, threshold, reference='window', cap=SNR_RULE_CAP):
    """
    Factors of the time weight of each link (t, t+1) of a series from the signal-to-noise ratio of its steps.

    A link is weighted by threshold_v / min(SNR_t, SNR_t+1), at least 1 and at most cap: a step whose data are below
    the threshold listens to its neighbours in proportion to how far below it is. With reference='window' the
    threshold is min(threshold, the best SNR of the series): a step is weak only next to a better one it can lean on,
    so a series in which every step is noisy is not tied harder than its data warrant. With reference='fixed' the
    threshold is used as it is. The threshold 12 used by default is 1/f, with f ~ 0.08 the fraction of power the Sun
    changes from one step to the next in the CRISP 8542 sequence (measured 2026-11-20).

    Parameters
    ----------
    snr : torch.Tensor
        (n_seq, n_t) ratio per step (torchmfbd.step_snr).
    threshold : float
        SNR below which a step leans on its neighbours.
    reference : str
        'window' or 'fixed'.
    cap : float
        Largest factor.

    Returns
    -------
    torch.Tensor
        (n_seq, n_t - 1) factors, all >= 1.
    """
    if reference not in ('window', 'fixed'):
        raise ValueError(f"snr_reference must be 'window' or 'fixed', got '{reference}'")
    snr = snr.clamp(min=1e-6)
    thr = torch.as_tensor(threshold, dtype=snr.dtype, device=snr.device)
    if reference == 'window':
        thr = torch.clamp(snr.amax(dim=1, keepdim=True), max=threshold)
    pair = torch.minimum(snr[:, 1:], snr[:, :-1])
    return torch.clamp(thr / pair, min=1.0, max=cap)
