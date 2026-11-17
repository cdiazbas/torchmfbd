.. include:: ../code_name
.. _lessons:


Choosing the settings: what was measured
========================================

This page collects what was learnt while testing |torchmfbd| against scenes with a known truth: synthetic bursts
(an object with a :math:`k^{-3}` power spectrum, Kolmogorov seeing, image motion and noise) and crops of the
CRISP 8542 sunspot of ``examples/spot_8542`` degraded with known seeing. Every number comes with the conditions it
was measured under; outside them it may not hold. The scenes and the metrics are in ``benchmarks/``, so any of these
results can be measured again, or measured for a new setting, with one command.

Errors are quoted as the rms of the difference between the object and the truth (seen through the diffraction limit),
divided by the standard deviation of the truth; a relative change of the error is quoted in percent. Unless stated
otherwise: LBFGS, 10 iterations, 12 frames.


Patch size and stride
---------------------

* **Use 128-pixel patches with a stride of 32 and an apodization border of 24 pixels**, as ``examples/spot_8542``
  now does. Against 64-pixel patches (stride 32, border 10), with the marginal loss: object error 13 to 23 % lower
  and PSF transfer error 29 to 45 % lower, in four sets of 31 scenes each (synthetic and real crops: standard,
  6 frames, twice the noise, and seeing that changes across the field). With the joint loss on a 512-pixel burst the
  run takes about three times longer (8 → 24 s on a V100S GPU).
* With 64-pixel patches the patch edges bias the fit: iterating longer makes the object worse (2 scenes, D/r0 5,
  joint loss: error 0.098 after 10 iterations, 0.139 after 30). With 96 to 128 pixels it does not.
* When the seeing changes across the field (anisoplanatism, a layer at 5 km), a stride of 64 is unsafe: with
  128-pixel patches the mean barely improves (−0.3 %) and 3 of 12 scenes come out more than 25 % worse (marginal
  loss). A stride of 32 holds.
* Larger patches do not help further: with a turbulent layer at 15 km, 160 and 192 pixels give −18.9 % and −15.8 %
  against −20.7 % for 128 pixels (joint loss, fixed prior).


Object filter: ``scharmer`` or ``garrote``
------------------------------------------

Measured with the joint loss only.

* With 64-pixel patches, ``garrote`` lowers the object error by 8.0 % (29 of 31 scenes), mostly above half the
  diffraction cutoff, and leaves the PSFs unchanged.
* With 128-pixel patches (stride 64) ``scharmer`` (Löfdahl–Scharmer) is better: −23.0 % against −20.8 % for
  ``garrote``, both relative to 64-pixel patches with ``scharmer``. It is also better when the true PSFs are given
  (0.121 against 0.124), so the difference is the filter, not the fit. The default stays ``scharmer``.
* With 6 frames and 64-pixel patches ``garrote`` makes no difference (−0.7 %).
* ``garrote`` changes the object smoothly when the data change slightly; the ``scharmer`` mask switches frequencies
  on or off, and a change of one part in :math:`10^4` in the frames can make the object jump.


What cannot be recovered
------------------------

* Between half and three quarters of the diffraction cutoff nothing is recovered: the error there is as large as the
  truth's own signal (0.99 to 1.00 of it), even when the true PSFs are given. With 64-pixel patches the code makes it
  worse by adding false power: the error in that band is three times the truth's signal, against once for 128-pixel
  patches (6 real scenes, D/r0 10, joint loss).


The marginal loss: when to stop fitting the object spectrum
------------------------------------------------------------

* ``stop_psd_optimization`` freezes the parameters of the object power spectrum after that many modes have been
  added. Freezing early gives PSFs that are too narrow; fitting to the end (as the code did before it honoured the
  option) gives PSFs that are too broad.
* Stopping halfway, at 20 modes (5 of the 10 outer steps of the example), was better than fitting to the end in 11
  of 13 synthetic scenes: object error 0.455 → 0.304, PSF error 0.465 → 0.439 (64-pixel patches).

.. TODO(check): repeat the stop_psd_optimization sweep with 128-pixel patches before stating it for the new example.


The joint loss: level of the object prior
-----------------------------------------

* With 96- and 128-pixel patches the best ``s_u_joint`` grows linearly with the noise and with the inverse of the
  number of frames: about :math:`250\,(\sigma/0.02)\,(12/N)` (measured at 96 pixels, checked at 128).
* With 64-pixel patches do not apply that rule: it makes the object worse than the fixed ``s_u_joint = 250``
  (+6.5 % in standard conditions, +9.2 % with 6 frames, +57 % with twice the noise; 31 scenes).


The optimizer
-------------

* LBFGS amplifies tiny changes: a change of one part in :math:`10^7` in the frames changes the object by 32 to 35 %
  (Adam: by :math:`10^{-6}`). The same configuration run on another machine gives a different result: the PSF error
  of one scene differed by up to 0.24 between a Mac and a Linux CPU, and by 3 to 4 % between two GPU types (V100S and
  T4). **Compare configurations only on the same machine**, and over many scenes.


Time series (``filter_loss: wiener_time``)
------------------------------------------

* Each time step is weighted by its own noise. A step ten times noisier than the rest no longer spoils the PSFs of
  the whole series (error of the quiet steps 1.17 → 0.46, against 0.45 with no noisy step); with twice the noise the
  effect is small (0.50 → 0.47) and with 1.3 times it is none.
* ``regularization.time.mask_window`` (default 3) computes the ``scharmer`` mask, or the ``garrote`` gain, of each
  step from the step and its neighbours. With 1, a badly degraded step keeps an error of 0.72; with 3 it comes back to
  the error of a clean step (0.19).
* With the best weight of the time term the first and last steps improve less than the middle ones, because they have
  one neighbour. Penalizing the curvature of the time evolution instead of the difference between steps does not
  change this (mean error 0.224 → 0.219) and makes a series with one bad step worse (0.202 → 0.214).


Judging a result without a truth
--------------------------------

* None of the measures that need no truth picks the best configuration better than simply using 128-pixel patches:
  over 241 scenes and 6 configurations, the configuration picked by the best of them (the Fourier ring correlation
  between two halves of the burst, combined with the :math:`\chi^2` at low frequencies) has on average an error 15.5 %
  above the best one, against 5.1 % for fixing 128-pixel patches.
* The :math:`\chi^2` at high frequencies misleads: the better the fit there, the worse the object (it fits the noise).
  And with 64-pixel patches, fitting longer lowers the loss while the object gets worse.
* For PSFs, the error of each frame's :math:`|{\rm OTF}|` does not predict how good the object will be (rank
  correlation −0.18); the transfer error :math:`R = \sum \hat T^* T / \sum |\hat T|^2` does (+0.71 on real crops,
  +0.88 on synthetic scenes), and is the one ``benchmarks/`` reports.


Tried and not adopted
---------------------

* **Restricting the wavefront to the directions the data determine** (``psf.nullspace_d_r0``, with
  ``psf.nullspace_threshold``; experimental, off by default, see :ref:`configuration`): it helps synthetic scenes
  (mean error 0.265 → 0.235 over 13 scenes, −11 %) but makes the real sunspot worse at D/r0 5 (+9 %, 6 of 6 scenes),
  with worse PSFs in 3 of 4 groups (64-pixel patches, joint loss). Leave it off unless your data are tested with
  ``benchmarks/``.
* A power-law object spectrum in the joint loss instead of the constant ``s_u_joint``: at best it ties the constant
  (0.268 against 0.267, better in 7 of 13 synthetic scenes); every other variant is worse (0.29 to 0.45) and gives
  worse PSFs.
* Weighting each frame by its own noise in the loss: the object gets worse (0.604 → 0.759 with Adam, 0.586 → 0.718
  with LBFGS; 32-pixel scenes with a different noise per frame, 5 scenes).
* A port of the ``scharmer`` filter to the GPU: the same masks, but slower (479 ms against 19 ms on an Apple GPU,
  30 patches) because its flood fill is sequential. ``garrote`` runs on the GPU and is up to 10 times faster than
  ``scharmer`` with many patches.
