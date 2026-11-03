import torchmfbd
import torch
from pathlib import Path

def test_marginal(make_synthetic_frames, object_error):

    n_seq = 4
    n_frames = 5
    sigma = 0.005

    config_path = Path(__file__).parent / 'marginal.yaml'

    obj, frames = make_synthetic_frames(config_path, n_seq, n_frames, sigma, seed=0)

    decSI = torchmfbd.Deconvolution(config_path)
    decSI.add_frames(frames, sigma=sigma * torch.ones(n_seq, n_frames), id_object=0, id_diversity=0, diversity=0.0)

    decSI.deconvolve(infer_object=False,
                     optimizer='adam',
                     simultaneous_sequences=200,
                     n_iterations=100)

    loss = decSI.loss
    assert torch.all(torch.isfinite(loss)), f'Non-finite loss at iterations {torch.where(~torch.isfinite(loss))[0].tolist()}'
    assert loss[-1] < loss[0], f'Loss did not decrease: {loss[0].item()} -> {loss[-1].item()}'

    assert decSI.obj[0].shape == obj.shape, f'Expected {obj.shape}, got {decSI.obj[0].shape}'
    assert torch.all(torch.isfinite(decSI.obj[0])), 'Non-finite object'

    err_obj, err_frame = object_error(decSI, obj, frames)
    assert torch.all(err_obj < err_frame), f'Object error {err_obj.tolist()} not below best frame error {err_frame.tolist()}'

    # Averaged over the sequences, 0.8 lies between the error with the fitted PSFs and the error with
    # the diffraction-limited PSF (power spectrum and noise still fitted), so the check also fails if
    # the wavefront is not fitted
    ratio = torch.mean(err_obj) / torch.mean(err_frame)
    assert ratio < 0.8, f'Mean object error is {ratio.item()} of the mean best frame error'

    # Fitted power spectrum exponent p (K, v0, p per object)
    p = decSI.pars_s0[0][:, 2]
    assert torch.all((p > 0.0) & (p < 10.0)), f'Fitted exponent p = {p.tolist()}'

    # Fitted noise variance per frequency; with the ortho FFT it is the variance of the pixel noise
    s2 = decSI.pars_s2[0][:, 0]
    assert torch.all((s2 > 0.5 * sigma**2) & (s2 < 2.0 * sigma**2)), f'Fitted noise variance {s2.tolist()}, injected {sigma**2}'

    # update_object with the cutoffs of the run gives back the run's object; a lower cutoff removes the
    # frequencies above it
    obj_run = decSI.obj[0].clone()
    decSI.update_object(cutoffs=[[0.95, 0.99]])
    change = torch.max(torch.abs(decSI.obj[0] - obj_run)) / torch.std(obj_run)
    assert change < 1e-5, f'update_object with the same cutoffs changed the object by {change.item():.2e} of its std'
