import pytest
import torch
import yaml
import torchmfbd

@pytest.fixture
def make_synthetic_frames():

    def _synthetic_frames(config, n_seq, n_frames, sigma, seed=0, amplitude=0.5, kmax=4, contrast=0.1, static=False):
        """
        Seeded synthetic bursts: a smooth periodic object of mean 1, blurred by known PSFs from
        Deconvolution.compute_psfs (KL modes with random coefficients) plus white Gaussian noise of
        standard deviation sigma. Returns the object (n_seq, n, n) and the frames (n_seq, n_frames, n, n).
        With static=True all the bursts see the first object (a Sun that does not change between time steps),
        each through its own seeing.
        """

        gen = torch.Generator().manual_seed(seed)

        # The data are generated on the CPU whatever device the configuration asks for
        if not isinstance(config, dict):
            with open(config) as f:
                config = yaml.safe_load(f)
        config['optimization']['gpu'] = -1
        dec = torchmfbd.Deconvolution(config)
        npix = dec.npix

        # Object band-limited to |k| <= kmax cycles per field, well inside the diffraction cutoff
        k = torch.fft.fftfreq(npix) * npix
        kx, ky = torch.meshgrid(k, k, indexing='ij')
        kk = torch.sqrt(kx**2 + ky**2)
        support = ((kk > 0) & (kk <= kmax)).float()
        re = torch.randn(n_seq, npix, npix, generator=gen)
        im = torch.randn(n_seq, npix, npix, generator=gen)
        obj = torch.fft.ifft2((re + 1j * im) * support / (1.0 + kk)).real
        obj = 1.0 + contrast * obj / torch.std(obj, dim=(-1, -2), keepdim=True)
        if static:
            obj = obj[:1].expand(n_seq, npix, npix).clone()

        # compute_psfs needs the basis and the diversity bookkeeping that deconvolve() sets up:
        # one object, one diversity channel starting at frame 0
        dec.n_o = 1
        dec.init_frame_diversity = [[0]]
        dec._define_basis()

        # Tip-tilt is left out: deconvolve() references it to its mean over the frames, so it would only shift the object
        modes = amplitude * torch.randn(n_seq, n_frames, dec.n_modes, generator=gen) / torch.sqrt(torch.arange(dec.n_modes) + 1.0)
        modes[:, :, 0:2] = 0.0
        psf, _ = dec.compute_psfs(modes, [torch.zeros(n_seq, n_frames)])

        # PSFs are centred at pixel (0, 0) and sum to 1, so the product of plain FFTs is the convolution
        frames = torch.fft.ifft2(torch.fft.fft2(obj)[:, None, :, :] * torch.fft.fft2(psf[0])).real
        frames = frames + sigma * torch.randn(frames.shape, generator=gen)

        return obj, frames

    return _synthetic_frames

@pytest.fixture
def object_error():

    def _object_error(dec, obj, frames):
        """
        rms error of the deconvolved object and of the best frame of each sequence with respect to the
        true object, all seen through the diffraction mask of the run and without the apodized border.
        """

        mask = dec.mask_diffraction_th[0].cpu()
        truth = torch.fft.ifft2(torch.fft.fft2(obj) * mask).real
        frames_filtered = torch.fft.ifft2(torch.fft.fft2(frames) * mask).real

        b = dec.npix_apod
        inner = (..., slice(b, -b), slice(b, -b))

        err_obj = torch.sqrt(torch.mean((dec.obj[0].cpu() - truth)[inner]**2, dim=(-1, -2)))
        err_frames = torch.sqrt(torch.mean((frames_filtered - truth[:, None, :, :])[inner]**2, dim=(-1, -2)))

        return err_obj, torch.min(err_frames, dim=1).values

    return _object_error
