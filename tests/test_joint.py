import torchmfbd
import torch
import yaml
from pathlib import Path

def test_joint(make_synthetic_frames, object_error):

    n_seq = 4
    n_frames = 5
    sigma = 0.005

    config_path = Path(__file__).parent / 'joint.yaml'

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

    # Averaged over the sequences, 0.75 lies between the error with the fitted PSFs and the error with
    # the diffraction-limited PSF, so the check also fails if the wavefront is not fitted
    ratio = torch.mean(err_obj) / torch.mean(err_frame)
    assert ratio < 0.75, f'Mean object error is {ratio.item()} of the mean best frame error'

def _time_config():

    config_path = Path(__file__).parent / 'joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['optimization']['filter_loss'] = 'wiener_time'
    config['regularization'] = {'time': {'lambda': 0.0}}

    return config_path, config

def test_time_4d(make_synthetic_frames):

    n_seq = 2
    n_frames = 5
    sigma = 0.005

    config_path, config = _time_config()

    obj, frames = make_synthetic_frames(config_path, n_seq, n_frames, sigma, seed=0)

    # A single time step per sequence (n_t = 1)
    decSI = torchmfbd.Deconvolution(config)
    decSI.add_frames(frames, sigma=sigma * torch.ones(n_seq, n_frames), id_object=0, id_diversity=0, diversity=0.0)

    decSI.deconvolve(infer_object=False,
                     optimizer='adam',
                     simultaneous_sequences=200,
                     n_iterations=20)

    assert torch.all(torch.isfinite(decSI.loss)), f'Non-finite loss at iterations {torch.where(~torch.isfinite(decSI.loss))[0].tolist()}'

    out = decSI.obj[0].cpu()
    assert out.shape == (n_seq, 32, 32), f'Expected {(n_seq, 32, 32)}, got {out.shape}'
    assert out.dtype == torch.float32, f'Expected float32, got {out.dtype}'
    assert torch.all(torch.isfinite(out)), 'Non-finite object'

    # The object keeps the mean level of the frames
    assert torch.allclose(out.mean(dim=(-1, -2)), frames.mean(dim=(-1, -2, -3)), rtol=0.05), f'Object mean {out.mean(dim=(-1, -2)).tolist()}, frames mean {frames.mean(dim=(-1, -2, -3)).tolist()}'

def test_time_5d(make_synthetic_frames):

    n_seq = 2
    n_frames = 5
    sigma = 0.005

    config_path, config = _time_config()

    obj, frames = make_synthetic_frames(config_path, n_seq, n_frames, sigma, seed=0)

    # The two sequences as two time steps of one sequence, (s, t, f, x, y)
    decSI = torchmfbd.Deconvolution(config)
    decSI.add_frames(frames[None, ...], sigma=sigma * torch.ones(1, n_seq, n_frames), id_object=0, id_diversity=0, diversity=0.0)

    decSI.deconvolve(infer_object=False,
                     optimizer='adam',
                     simultaneous_sequences=200,
                     n_iterations=20)

    assert torch.all(torch.isfinite(decSI.loss)), f'Non-finite loss at iterations {torch.where(~torch.isfinite(decSI.loss))[0].tolist()}'

    out = decSI.obj[0].cpu()
    assert out.shape == (1, n_seq, 32, 32), f'Expected {(1, n_seq, 32, 32)}, got {out.shape}'
    assert out.dtype == torch.float32, f'Expected float32, got {out.dtype}'
    assert torch.all(torch.isfinite(out)), 'Non-finite object'
    assert torch.allclose(out.mean(dim=(-1, -2)), frames.mean(dim=(-1, -2, -3))[None, :], rtol=0.05), f'Object mean {out.mean(dim=(-1, -2)).tolist()}, frames mean {frames.mean(dim=(-1, -2, -3)).tolist()}'

def test_time_contrast_weights_positive(make_synthetic_frames, monkeypatch):
    # wiener_time_contrast weights the time term by the contrast of each step. It used to divide by the zero frequency
    # of the apodized patches, which remove_gradient_apodization takes to ~0: contrasts of either sign, which can turn
    # the time weight negative. The step with the sharper frames must get the larger contrast.
    config_path, config = _time_config()
    config['optimization']['filter_loss'] = 'wiener_time_contrast'
    config['regularization'] = {'time': {'lambda': 1e-3}}
    config['images']['remove_gradient_apodization'] = True
    obj, frames = make_synthetic_frames(config_path, 2, 5, 0.005, seed=0)
    blurred = frames.clone()
    blurred[1] = torch.nn.functional.avg_pool2d(frames[1], 3, 1, 1, count_include_pad=False)    # step 1 less sharp
    seen = []
    orig = torchmfbd.Deconvolution.solve_object_time_evolution
    def spy(self, S, D, Reg_Space, lambda_s, lambda_t, contrast=None):
        seen.append(contrast.detach().cpu().clone())
        return orig(self, S, D, Reg_Space, lambda_s, lambda_t, contrast=contrast)
    monkeypatch.setattr(torchmfbd.Deconvolution, 'solve_object_time_evolution', spy)
    dec = torchmfbd.Deconvolution(config)
    dec.add_frames(blurred[None, ...], sigma=0.005 * torch.ones(1, 2, 5), id_object=0, id_diversity=0, diversity=0.0)
    dec.deconvolve(infer_object=False, optimizer='adam', simultaneous_sequences=200, n_iterations=3)
    assert seen and all(torch.all(torch.isfinite(c)) and torch.all(c > 0) for c in seen)
    assert all(c[0] > c[1] for c in seen)
    assert torch.all(torch.isfinite(dec.obj[0]))

def test_time_loss_skips_the_mask(make_synthetic_frames, monkeypatch):
    # The loss of the time path uses only the unfiltered object, so it does not compute the scharmer mask
    # (need_filter=False). The fit must be the same as when the mask is computed and thrown away.
    def fit(force_mask):
        config_path, config = _time_config()
        config['regularization'] = {'time': {'lambda': 1e-3}}
        obj, frames = make_synthetic_frames(config_path, 2, 5, 0.005, seed=0)
        calls = []
        orig_cobj, orig_ls = torchmfbd.Deconvolution.compute_object, torchmfbd.Deconvolution.lofdahl_scharmer_filter
        def cobj(self, *a, **k):
            if force_mask:
                k['need_filter'] = True
            return orig_cobj(self, *a, **k)
        def ls(self, *a, **k):
            calls.append(1)
            return orig_ls(self, *a, **k)
        monkeypatch.setattr(torchmfbd.Deconvolution, 'compute_object', cobj)
        monkeypatch.setattr(torchmfbd.Deconvolution, 'lofdahl_scharmer_filter', ls)
        dec = torchmfbd.Deconvolution(config)
        dec.add_frames(frames[None, ...], sigma=0.005 * torch.ones(1, 2, 5), id_object=0, id_diversity=0, diversity=0.0)
        dec.deconvolve(infer_object=False, optimizer='lbfgs', simultaneous_sequences=200, n_iterations=3)
        monkeypatch.undo()
        return dec, len(calls)
    fast, n_fast = fit(False)
    full, n_full = fit(True)
    assert torch.equal(fast.obj[0], full.obj[0]) and torch.equal(fast.loss, full.loss)
    assert n_fast < n_full

def test_prior_frames_scale():

    # The joint object prior keeps its weight relative to the data above 12 frames (s_u_joint * 12 / N) and is
    # unchanged at 12 frames or fewer; s_u_joint is stated for 64-px patches (here 32 px: factor (32/64)^2)
    config_path = Path(__file__).parent / 'joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['optimization']['gpu'] = -1
    config['images']['remove_fixed_pattern'] = False

    s_u_joint = config['object1']['s_u_joint']
    for n_frames, expected in ((5, 1.0), (12, 1.0), (24, 0.5), (48, 0.25)):
        dec = torchmfbd.Deconvolution(config)
        dec.add_frames(1.0 + 0.01 * torch.rand(1, n_frames, 32, 32), sigma=0.01, id_object=0, id_diversity=0, diversity=0.0)
        dec.n_o = 1
        dec.init_frame_diversity = [[0]]
        dec._define_basis()
        dec.combine_frames()
        s_u, _, _, _, _ = dec.get_su_s2(obj=0, sigma=[0.01 * torch.ones(1, n_frames)])
        value = float(s_u.flatten()[1])
        assert abs(value - s_u_joint * 0.25 * expected) < 1e-6 * s_u_joint, f'{n_frames} frames: s_u = {value}, expected {s_u_joint * 0.25 * expected}'

def test_object_independent_of_batch_noise():

    # A sequence's object must not depend on the noise of the other sequences deconvolved in the same batch:
    # sequence A (sigma 0.01) alone and together with a noisy sequence B (sigma 0.1), true OTFs (BUG-170)
    config_path = Path(__file__).parent / 'joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['optimization']['gpu'] = -1
    config['images']['remove_fixed_pattern'] = False

    gen = torch.Generator().manual_seed(3)
    n_frames = 6
    for image_filter in ('scharmer', 'garrote', 'tophat'):
        dec = torchmfbd.Deconvolution(config)
        dec.n_o = 1
        dec.n_t = 1
        dec.init_frame_diversity = [[0]]
        dec.image_filter = [image_filter]
        dec.loss_filter = 'wiener'
        dec._define_basis()
        dec.compute_diffraction_masks()
        npix = dec.npix

        obj = 1.0 + 0.1 * torch.randn(2, npix, npix, generator=gen)
        modes = 0.4 * torch.randn(2, n_frames, dec.n_modes, generator=gen) / torch.sqrt(torch.arange(dec.n_modes) + 1.0)
        modes[:, :, 0:2] = 0.0
        with torch.no_grad():
            _, otf = dec.compute_psfs(modes, [torch.zeros(2, n_frames)])
        sigma = torch.tensor([[0.01] * n_frames, [0.1] * n_frames])
        noise = torch.randn(2, n_frames, npix, npix, generator=gen) * sigma[:, :, None, None]
        frames_ft = npix * torch.fft.fft2(obj, norm='ortho')[:, None] * otf[0] + torch.fft.fft2(noise, norm='ortho')

        with torch.no_grad():
            _, _, both = dec.compute_object([frames_ft], [otf[0]], [sigma], None)
            _, _, alone = dec.compute_object([frames_ft[0:1]], [otf[0][0:1]], [sigma[0:1]], None)

        diff = float(torch.max(torch.abs(both[0][0] - alone[0][0])) / torch.std(alone[0][0]))
        assert diff < 1e-4, f'{image_filter}: the object of sequence A changes by {diff:.3e} std when a noisy sequence shares its batch'

def test_jitter_with_threaded_fft(make_synthetic_frames):

    # The jitter OTF multiplies the OTF returned by the threaded FFT, a custom autograd function: that product
    # must not be in place (BUG-172). The threaded FFT is used on CPU builds without MKL when the batch has
    # at least 4 images per thread, so two threads and 2 x 5 frames reach it
    config_path = Path(__file__).parent / 'joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['optimization']['gpu'] = -1
    config['images']['remove_fixed_pattern'] = False
    config['annealing']['type'] = 'none'

    obj, frames = make_synthetic_frames(config_path, 2, 5, 0.005, seed=0)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for jitter in ('slow', 'fast'):
            config['psf']['jitter'] = jitter
            dec = torchmfbd.Deconvolution(config)
            dec.add_frames(frames, sigma=0.005 * torch.ones(2, 5), id_object=0, id_diversity=0, diversity=0.0)
            dec.deconvolve(infer_object=False, optimizer='adam', simultaneous_sequences=2, n_iterations=2)
            assert torch.all(torch.isfinite(dec.obj[0])), f'jitter {jitter}: non-finite object'
    finally:
        torch.set_num_threads(n_threads)

def test_psf_fit_independent_of_batch_noise(make_synthetic_frames):

    # The joint loss must give each sequence its own noise variance: the PSFs and objects fitted to two quiet
    # sequences may not change when the two other sequences of the batch are noisy (BUG-170, loss)
    config_path = Path(__file__).parent / 'joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['optimization']['gpu'] = -1
    config['images']['remove_fixed_pattern'] = False

    obj, frames = make_synthetic_frames(config_path, 4, 5, 0.01, seed=1)
    gen = torch.Generator().manual_seed(7)
    extra = torch.randn(frames.shape, generator=gen)
    out = []
    for noisy in (False, True):
        sigma = 0.01 * torch.ones(4, 5)
        fr = frames.clone()
        if noisy:
            sigma[2:] = 0.1
            fr[2:] = fr[2:] + 0.0995 * extra[2:]
        torch.manual_seed(0)
        dec = torchmfbd.Deconvolution(config)
        dec.add_frames(fr, sigma=sigma, id_object=0, id_diversity=0, diversity=0.0)
        dec.deconvolve(infer_object=False, optimizer='adam', simultaneous_sequences=4, n_iterations=10)
        out.append(dec.obj[0][0:2].detach().double())
    diff = float(torch.max(torch.abs(out[1] - out[0])) / torch.std(out[0]))
    assert diff < 1e-3, f'the objects of the quiet sequences change by {diff:.3e} std when the other sequences of the batch are noisy'

def test_time_mask_from_neighbouring_steps():

    # In the time path the L&S mask of a time step is computed from that step and its neighbours, so a step with
    # ten times the noise of the others keeps what the time term brings from them (BUG-171). True OTFs: only the
    # object solver is tested
    config_path = Path(__file__).parent / 'joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['optimization']['gpu'] = -1
    config['images']['remove_fixed_pattern'] = False

    gen = torch.Generator().manual_seed(4)
    n_seq, n_t, n_frames = 2, 6, 6
    dec = torchmfbd.Deconvolution(config)
    dec.n_o = 1
    dec.n_t = n_t
    dec.init_frame_diversity = [[0]]
    dec.image_filter = ['scharmer']
    dec.loss_filter = 'wiener_time'
    dec.time_regularization = 5e-4
    dec._define_basis()
    dec.compute_diffraction_masks()
    npix = dec.npix

    # An object that changes slowly from one time step to the next
    k = torch.fft.fftfreq(npix)
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    amp = 1.0 / (1.0 + (torch.sqrt(kx**2 + ky**2) / 0.08)**2)
    def field():
        f = torch.fft.ifft2(torch.fft.fft2(torch.randn(n_seq, npix, npix, generator=gen)) * amp).real
        return 0.1 * f / torch.std(f, dim=(-1, -2), keepdim=True)
    a, b = field(), field()
    th = 0.1 * torch.arange(n_t, dtype=torch.float32)[None, :, None, None]
    obj = (1.0 + torch.cos(th) * a[:, None] + torch.sin(th) * b[:, None]).reshape(n_seq * n_t, npix, npix)

    modes = torch.zeros(n_seq * n_t, n_frames, dec.n_modes)
    modes[..., 2:20] = 0.4 * torch.randn(n_seq * n_t, n_frames, 18, generator=gen)
    with torch.no_grad():
        _, otf = dec.compute_psfs(modes, [torch.zeros(n_seq * n_t, n_frames)])
    sigma = torch.full((n_seq, n_t, n_frames), 0.01)
    sigma[:, 3] = 0.1
    sigma = sigma.reshape(n_seq * n_t, n_frames)
    noise = torch.randn(n_seq * n_t, n_frames, npix, npix, generator=gen) * sigma[:, :, None, None]
    frames_ft = npix * torch.fft.fft2(obj, norm='ortho')[:, None] * otf[0] + torch.fft.fft2(noise, norm='ortho')

    with torch.no_grad():
        _, _, out = dec.compute_object([frames_ft.to(torch.complex64)], [otf[0].to(torch.complex64)], [sigma], None)

    ref = torch.fft.ifft2(torch.fft.fft2(obj) * dec.mask_diffraction_th[0]).real
    err = torch.sqrt(torch.mean((out[0].double() - ref.double())**2, dim=(-1, -2))) / torch.std(ref.double(), dim=(-1, -2))
    err = err.reshape(n_seq, n_t).mean(dim=0)
    others = torch.cat([err[:3], err[4:]]).mean()
    assert err[3] < 1.5 * others, f'noisy step: error {float(err[3]):.3f} against {float(others):.3f} for the other steps'

def test_time_fit_robust_to_one_noisy_step():

    # The time-coupled loss must weigh each time step by its own noise: one step with ten times the noise of the
    # others may not spoil the PSFs, and so the objects, fitted to the quiet steps (BUG-170, time loss). Blind fit;
    # the quiet steps are compared with a control in which every step is quiet
    config_path, config = _time_config()
    config['optimization']['gpu'] = -1
    config['images']['remove_fixed_pattern'] = False
    config['regularization']['time']['lambda'] = 1e-3

    n_seq, n_t, n_frames = 2, 6, 6
    gen = torch.Generator().manual_seed(20)
    dec0 = torchmfbd.Deconvolution(config)
    dec0.n_o = 1
    dec0.init_frame_diversity = [[0]]
    dec0._define_basis()
    npix = dec0.npix

    k = torch.fft.fftfreq(npix)
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    amp = 1.0 / (1.0 + (torch.sqrt(kx**2 + ky**2) / 0.08)**2)
    def field():
        f = torch.fft.ifft2(torch.fft.fft2(torch.randn(n_seq, npix, npix, generator=gen)) * amp).real
        return 0.1 * f / torch.std(f, dim=(-1, -2), keepdim=True)
    a, b = field(), field()
    th = 0.1 * torch.arange(n_t, dtype=torch.float32)[None, :, None, None]
    obj = 1.0 + torch.cos(th) * a[:, None] + torch.sin(th) * b[:, None]
    modes = torch.zeros(n_seq * n_t, n_frames, dec0.n_modes)
    modes[..., 2:20] = 0.4 * torch.randn(n_seq * n_t, n_frames, 18, generator=gen)
    with torch.no_grad():
        psf, _ = dec0.compute_psfs(modes, [torch.zeros(n_seq * n_t, n_frames)])
    psf = psf[0].reshape(n_seq, n_t, n_frames, npix, npix)
    clean = torch.fft.ifft2(torch.fft.fft2(obj)[:, :, None] * torch.fft.fft2(psf)).real
    noise = torch.randn(clean.shape, generator=gen)
    quiet = [t for t in range(n_t) if t != 3]

    err = {}
    for case in ('control', 'noisy step'):
        sigma = torch.full((n_seq, n_t, n_frames), 0.01)
        if case == 'noisy step':
            sigma[:, 3] = 0.1
        torch.manual_seed(0)
        dec = torchmfbd.Deconvolution(config)
        dec.add_frames(clean + noise * sigma[..., None, None], sigma=sigma, id_object=0, id_diversity=0, diversity=0.0)
        dec.deconvolve(infer_object=False, optimizer='adam', simultaneous_sequences=n_seq * n_t, n_iterations=40)
        ref = torch.fft.ifft2(torch.fft.fft2(obj) * dec.mask_diffraction_th[0].cpu()).real.double()
        o = dec.obj[0].cpu().double().reshape(n_seq, n_t, npix, npix)
        d = (o - o.mean(dim=(-1, -2), keepdim=True)) - (ref - ref.mean(dim=(-1, -2), keepdim=True))
        e = torch.sqrt(torch.mean(d**2, dim=(-1, -2))) / torch.std(ref, dim=(-1, -2))
        err[case] = float(e[:, quiet].mean())

    assert err['noisy step'] < 1.3 * err['control'], f"quiet steps: error {err['noisy step']:.3f} with a noisy step against {err['control']:.3f} without"
