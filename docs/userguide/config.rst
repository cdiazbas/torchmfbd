.. include:: ../code_name
.. _configuration:


Configuration file
==================

Configuration files are YAML human-readable files. Note that some of them are optional
and can be absent from the configuration file. If this is the case, a default
value will be used.

Spatially invariant
-------------------

::

    telescope:
        diameter: 100.0
        central_obscuration : 0.0
        spider : 0

    images:
        n_pixel : 64
        pix_size : 0.059
        apodization_border : 6

    object1:
        wavelength : 8542.0
        image_filter: tophat
        cutoff : [0.75, 0.80]
        s_u_joint : 100.0

    object2:
        wavelength : 8542.0
        image_filter: tophat
        cutoff : [0.75, 0.80]
        s_u_joint : 100.0
        
    optimization:
        gpu : 0
        transform : softplus
        softplus_scale : 1.0        
        lr_obj : 0.02
        lr_modes : 0.08
        loss_type : joint
        show_object_info: False

    regularization:
        iuwt1:
            variable : object
            lambda : 0.0
            nbands : 5

    psf:
        model : kl
        nmax_modes : 44
        orthogonalize : False

    initialization:
        object : contrast
        modes_std : 0.0

    annealing:
        type: linear
        start_pct : 0.0
        end_pct : 0.85

telescope
^^^^^^^^^

* ``diameter``: diameter of the telescope in cm
* ``central_obscuration`` : central obscuration of the telescope in cm
* ``spider`` : width of the spider in cm

images
^^^^^^

* ``n_pixel`` : size in pixels of the images to be deconvolved. In case the images are mosaicked to take into account the size of the anisoplanatic patch, this keyword gives the size of the patches
* ``pix_size`` : pixel size in arcsec
* ``apodization_border`` : border of the apodization in pixels to avoid edge effects in the Fourier transform used for the convolution
* ``remove_fixed_pattern`` : if True (the default), a pattern fixed on the detector is removed from each burst passed to ``add_frames``: the frequencies above 0.75 of the diffraction cutoff where the frames agree with each other (coherence across frames, which solar structure near the cutoff does not have) are taken as pattern and their mean over frames is subtracted. Such a pattern is not blurred by the telescope, so the deconvolution would otherwise amplify it near the cutoff into stripes. It needs at least 12 frames per burst (it is skipped, with a warning, below that) and frames in detector coordinates: on registered or destretched frames the pattern moves and is not found. The patterns found are kept in ``fixed_pattern``; ``torchmfbd.util.remove_fixed_pattern`` applies it to any burst.

objects
^^^^^^^

Since several objects can be deconvolved simultaneously, one needs to define the parameters for each object. Objects are defined as ``objectx``, where ``x`` is an index
starting from 1.

* ``wavelength`` : wavelength of the object in angstroms
* ``image_filter`` : Fourier filter to be used for the image. Options are ``tophat`` (a simple tophat filter to avoid frequencies above the cutoff), ``scharmer`` (the filter from Lofdahl & Scharmer 1994) or ``garrote`` (the gain of the Lofdahl & Scharmer filter used as a continuous weight, with the threshold dropped and the flood fill replaced by a grayscale reconstruction from zero frequency, so that the object changes continuously with the data).
* ``cutoff`` : cutoff frequency. The first element indicates the cutoff frequency below which the filter is 1. The second element marks the frequency above which the filter is 0. In between, a smooth transition is built. All are given in units of the diffraction limit.
* ``s_u_joint`` : Estimation of the power spectrum of the object, for 64-pixel patches; it is scaled with the square of ``n_pixel`` internally, so the same value means the same regularization at every patch size. With more than 12 frames per object it is also multiplied by 12/N, so that the prior keeps its weight relative to the data term, which grows with the number of frames N. This is relevant only if the ``joint`` loss is used. With ``auto`` the level follows the data: 250 (sigma / mu / 0.02) max(12 / N, 1) mu^2, with sigma the noise estimate of the frames, mu their mean and N the frames per sequence (for frames divided by their mean, 250 (sigma / 0.02) max(12 / N, 1)). On the benchmark with 128-px patches, together with the ``coupling`` regularization, it lowers the object error by 10 % against the marginal loss with 12 and with 6 frames; without it the joint loss fails on 6-frame bursts. With 64-px patches it makes the object worse

optimization
^^^^^^^^^^^^
* ``gpu`` : index of the GPU to be used for the optimization. If you want to use the CPU, just put ``-1``.
* ``transform`` : in case the object is optimized simultaneously with the wavefronts, a transformation can be applied to ensure its positivity. Options are ``softplus`` (which uses the softplus function) or ``none`` (which uses the identity function).
* ``softplus_scale`` : scale of the softplus function
* ``lr_obj`` : learning rate for the object
* ``lr_modes`` : learning rate for the modes
* ``half_spectrum`` : ``true`` (the default) or ``false``. With ``true`` the joint loss with the ``wiener`` filter_loss is computed on half of the spectrum (``rfft2``): the frames and the PSFs are real, so the other half carries the same terms. The loss and its gradient are the same to float32 rounding; the fit of the example's 8542 burst takes 4.75 s instead of 5.53 s on a V100S. Not used with jitter, the marginal loss, the time filters or an object regularization
* ``show_object_info`` : if True, the object information (contrast, minimum and maximum) is shown during the optimization
* ``loss_type`` : type of loss to be used. Options are ``joint`` (which uses a joint loss on the object and the modes) or ``marginal`` (which uses the marginal loss on the modes only)

regularization
^^^^^^^^^^^^^^
Different regularizations can be used simultaneously. Current options are ``smooth`` (an L2 regularization applied on the spatial derivative of the images) or ``iuwt`` (an L1 regularization on the isotropic undecimated wavelet transform).

* ``iuwt1`` : iuwt regularization to be used. The parameters are:
    * ``variable`` : variable to be regularized. Options are ``object`` (which regularizes the object) or ``modes`` (which regularizes the modes)
    * ``lambda`` : regularization parameter
    * ``nbands`` : number of bands for the iuwt

* ``smooth1`` : iuwt regularization to be used. The parameters are:
    * ``variable`` : variable to be regularized. Options are ``object`` (which regularizes the object) or ``modes`` (which regularizes the modes)
    * ``lambda`` : regularization parameter    

* ``time`` : temporal regularization of the object, used with ``filter_loss: wiener_time`` or ``wiener_time_contrast`` on 5-D data (sequences, time steps, frames). The parameters are:
    * ``lambda`` : weight of the squared difference between consecutive time steps, for 64-pixel patches and 12 frames per time step; it is scaled internally so that the same value means the same smoothing for any patch size and number of frames. On a real solar sequence degraded with known seeing, 5e-4 to 1.5e-3 is best; 1.5e-4 is too weak with poor seeing (see :ref:`lessons`)
    * ``mask_window`` : number of consecutive time steps (odd, default 3) from which the ``scharmer`` mask or the ``garrote`` gain of each step is computed. With 1 each step uses only its own frames, and a much noisier step then loses what the time term brings from its neighbours
    * ``snr_threshold`` : SNR rule of the time weight, only with ``wiener_time``. The signal-to-noise ratio of each step is measured from its own frames (``torchmfbd.step_snr``: power between 0.1 and 0.3 of the diffraction cutoff over the noise power beyond 1.1 of the cutoff), and each link (t, t+1) gets ``lambda`` times ``threshold / min(SNR_t, SNR_t+1)``, at least 1 and at most 100. A step whose data are below the threshold then leans on its neighbours in proportion, instead of fitting a too sharp PSF to its noise. 12 is a constant, not a property of the series: on synthetic bursts whose Sun changes 0.024 to 0.16 of its power per step, thresholds 6 and 48 moved the error by −3.6 to +6.4 % against 12 with no trend in that change, and with one very bad step (D/r0 20) any threshold from 6 to 48 restored it alike while switching the rule off (threshold 1) doubled its error in one series (2026-11-23). On by default with 12; 0 gives one weight for every link, bit for bit as before the rule. Measured on 49-step series with real solar evolution as the truth (2026-11-21): never worse than one weight in 19 series, 7 to 33 % better in 16, most with very bad steps; against the time example as it was (windows of 7 every 4), better in 12 of 12 new series (mean 23.7 %), and on the real sequence it restores the very bad step as the reference restoration does
    * ``snr_reference`` : ``window`` (default) or ``fixed``. With ``window`` the threshold is the smaller of ``snr_threshold`` and the best SNR of the series being fitted, so a step is weighted up only next to a better one it can lean on; with ``fixed`` it is ``snr_threshold`` itself. With every step noisy (noise 0.04, frame SNR ~1.6) ``fixed`` ties the whole series and is 7.5 % worse than one weight, ``window`` is as good as one weight
    * ``snr_source`` : ``frames`` (default) or ``power``. ``frames`` measures each step's SNR from its frames, signal at 0.1-0.3 of the cutoff against the noise beyond it. Restored objects have no noise beyond the cutoff; for them ``power`` scores a step by its power at 0.1-0.3 of the cutoff relative to the median step of the series, which sits at ``snr_threshold``. ``torchmfbd.repair_series`` uses it when it repairs a series restored without time term with ``method='deconvolve'``

* ``coupling`` : couples the wavefronts of neighbouring patches. Neighbouring patches see almost the same low-order seeing, so the loss gets ``lambda`` times the sum over pairs of neighbours and frames of the squared differences of their first ``n_modes`` wavefront coefficients (tip-tilt referred to the first frame), normalised like the joint loss. The patches may still differ where the data ask for it. It needs the pairs of neighbours, given with ``Deconvolution.set_neighbours(patchify.neighbours())`` after ``Patchify4D.patchify``, and all patches fitted in one batch (``simultaneous_sequences`` at least the number of patches); only for the ``kl`` and ``zernike`` models. With a time dimension each pair is coupled within every time step, not across steps. With the experimental ``nullspace_d_r0`` the coupling acts on the unrestricted coefficients while the PSFs use the restricted ones (that pair is not measured). Absent by default. The parameters are:
    * ``lambda`` : weight of the term. Without the time term, 30 with stride 32 and 100 with stride 64 were the best values measured: neighbours further apart need a stronger coupling. With the time term (``wiener_time``) 30 is also best with stride 64: 100 was 1.4 to 4.2 % worse on 6 synthetic series (2026-11-23)
    * ``n_modes`` : number of wavefront coefficients coupled (default 10). Coupling 20 instead of 10 did not change the result

psf
^^^
* ``model`` : model to be used for the PSF. Options are ``kl`` (which uses Karhunen-Loeve modes) or ``zernike`` (which uses Zernike modes)
* ``nmax_modes`` : number of modes to be used for the PSF
* ``orthogonalize`` : if True, the modes are re-orthogonalized. This is relevant when strong central obscuration is present.
* ``nullspace_d_r0`` : if present, the fit is done twice. From the first, the information that the data of each patch carry on its wavefront coefficients is compared with a Kolmogorov prior of this D/r0 (the KL variances of the basis), and the second fit may change the wavefront only along the combinations of coefficients that the data determine better than the prior; the others stay at zero. Only for the ``kl`` model, the ``joint`` loss, one object and no phase diversity. It roughly doubles the time. Absent by default, and experimental: measured on 31 scenes with a known truth (64-px patches, LBFGS 10), it lowers the object error of synthetic bursts by 11 % but raises it on real sunspot scenes at D/r0 5 by 9 % (6 of 6 scenes), and the PSF error with it
* ``nullspace_threshold`` : ratio of data information to prior above which a direction is kept (default 1)

initialization
^^^^^^^^^^^^^^
* ``object`` : initialization of the object. Options are ``contrast`` (which initializes the object with the contrast) or ``random`` (which initializes the object with random values). This has impact only if the ``l2``loss is used.
* ``modes_std`` : standard deviation of the modes at initialization

annealing
^^^^^^^^^

The modes are annealed during the optimization. This means modes are added slowly while the optimization proceeds. The specific annealing :

* ``type`` : type of annealing. Options are ``sigmoid`` (which uses a sigmoid function), ``linear`` (which uses a linear function) or ``none``
* ``start_pct`` : percentage of the total number of iterations at which the annealing starts
* ``end_pct`` : percentage of the total number of iterations at which the annealing ends


Spatially variant
-------------------

::

    telescope:
        diameter: 100.0
        central_obscuration : 0.0

    images:
        n_pixel : 256
        wavelength : 8542.0
        pix_size : 0.059
        apodization_border : 6
        remove_gradient_apodization : False

    object1:
        wavelength : 8542.0
        image_filter: tophat
        cutoff : 0.75

    object2:
        wavelength : 8542.0
        image_filter: tophat
        cutoff : 0.75
            
    optimization:
        gpu : 0
        transform : softplus
        softplus_scale : 1000.0        
        lr_obj : 0.02
        lr_modes : 0.08
        lr_tt : 0.003

    regularization:
        smooth1:
            variable: tiptilt
            lambda: 0.01
        smooth2:
            variable: modes
            lambda: 0.01
        iuwt1:
            variable : object
            lambda : 0.00
            nbands : 5

    psf:
        model : nmf        
        nmax_modes : 150
        filename: '../basis/nmf_8542_n_150_r0_5_30.npz'
        ngrid_modes: 8

    initialization:
        object : contrast
        modes_std : 0.0

    annealing:
        type: none
        start_pct : 0.0
        end_pct : 0.85

telescope
^^^^^^^^^

* ``diameter``: diameter of the telescope in cm
* ``central_obscuration`` : central obscuration of the telescope in cm

images
^^^^^^

* ``n_pixel`` : size in pixels of the images to be deconvolved. In case the images are mosaicked to take into account the size of the anisoplanatic patch, this keyword gives the size of the patches
* ``pix_size`` : pixel size in arcsec
* ``apodization_border`` : border of the apodization in pixels to avoid edge effects in the Fourier transform used for the convolution
* ``remove_fixed_pattern`` : if True (the default), a pattern fixed on the detector is removed from each burst passed to ``add_frames``: the frequencies above 0.75 of the diffraction cutoff where the frames agree with each other (coherence across frames, which solar structure near the cutoff does not have) are taken as pattern and their mean over frames is subtracted. Such a pattern is not blurred by the telescope, so the deconvolution would otherwise amplify it near the cutoff into stripes. It needs at least 12 frames per burst (it is skipped, with a warning, below that) and frames in detector coordinates: on registered or destretched frames the pattern moves and is not found. The patterns found are kept in ``fixed_pattern``; ``torchmfbd.util.remove_fixed_pattern`` applies it to any burst.
* ``remove_gradient_apodization`` : if True, the gradient of the image is removed before the apodization. This is useful to remove artifacts when the images have strong gradients

objects
^^^^^^^

Since several objects can be deconvolved simultaneously, one needs to define the parameters for each object. Objects are defined as ``objectx``, where ``x`` is an index
starting from 1.

* ``wavelength`` : wavelength of the object in angstroms
* ``image_filter`` : Fourier filter to be used for the image. Options are ``tophat`` (a simple tophat filter to avoid frequencies above the cutoff), ``scharmer`` (the filter from Lofdahl & Scharmer 1994) or ``garrote`` (the gain of the Lofdahl & Scharmer filter used as a continuous weight, with the threshold dropped and the flood fill replaced by a grayscale reconstruction from zero frequency, so that the object changes continuously with the data).
* ``cutoff`` : cutoff frequency indicated in units of the diffraction limit

optimization
^^^^^^^^^^^^
* ``gpu`` : index of the GPU to be used for the optimization. If you want to use the CPU, just put ``-1``.
* ``transform`` : in case the object is optimized simultaneously with the wavefronts, a transformation can be applied to ensure its positivity. Options are ``softplus`` (which uses the softplus function) or ``none`` (which uses the identity function).
* ``softplus_scale`` : scale of the softplus function
* ``lr_obj`` : learning rate for the object
* ``lr_tt`` : learning rate for the tip-tilt
* ``lr_modes`` : learning rate for the modes
* ``half_spectrum`` : ``true`` (the default) or ``false``. With ``true`` the joint loss with the ``wiener`` filter_loss is computed on half of the spectrum (``rfft2``): the frames and the PSFs are real, so the other half carries the same terms. The loss and its gradient are the same to float32 rounding; the fit of the example's 8542 burst takes 4.75 s instead of 5.53 s on a V100S. Not used with jitter, the marginal loss, the time filters or an object regularization

regularization
^^^^^^^^^^^^^^
Different regularizations can be used simultaneously. Current options are ``smooth`` (an L2 regularization applied on the spatial derivative of the images) or ``iuwt`` (an L1 regularization on the isotropic undecimated wavelet transform).

* ``iuwt1`` : iuwt regularization to be used. The parameters are:
    * ``variable`` : variable to be regularized. Options are ``object`` (which regularizes the object) or ``modes`` (which regularizes the modes)
    * ``lambda`` : regularization parameter
    * ``nbands`` : number of bands for the iuwt

* ``smooth1`` : iuwt regularization to be used. The parameters are:
    * ``variable`` : variable to be regularized. Options are ``object`` (which regularizes the object) or ``modes`` (which regularizes the modes)
    * ``lambda`` : regularization parameter    

psf
^^^
* ``model`` : model to be used for the PSF. Options are ``nmf`` (which uses non-negative matrix factorization) or ``pca`` (which uses principal component analysis)
* ``nmax_modes`` : number of modes to be used for the PSF
* ``filename`` : filename of the basis to be used for the PSF
* ``ngrid_modes`` : number of grid points where to infer the PSF modes. They will be interpolated to the full size of the image

initialization
^^^^^^^^^^^^^^
* ``object`` : initialization of the object. Options are ``contrast`` (which initializes the object with the contrast) or ``random`` (which initializes the object with random values). This has impact only if the object is inferred.
* ``modes_std`` : standard deviation of the modes at initialization

annealing
^^^^^^^^^

The modes are annealed during the optimization. This means modes are added slowly while the optimization proceeds. The specific annealing :

* ``type`` : type of annealing. Options are ``sigmoid`` (which uses a sigmoid function), ``linear`` (which uses a linear function) or ``none``
* ``start_pct`` : percentage of the total number of iterations at which the annealing starts
* ``end_pct`` : percentage of the total number of iterations at which the annealing ends