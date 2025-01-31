import torchmfbd

tmp = torchmfbd.NMF(n_pixel=128,
                 wavelength=8542.0,
                 diameter=100.0,
                 pix_size=0.059,
                 central_obs=0.0,
                 n_modes=25,
                 r0_min=15.0,
                 r0_max=50.0)
    
# psf, modes, r0 = tmp.compute_psfs(100)
tmp.compute_nmf(100)