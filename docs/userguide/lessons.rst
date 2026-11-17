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
divided by the standard deviation of the truth; a relative change of the error is quoted in percent.

.. TODO(check, UiO): every number below against bench out*/ and rerun_uio/ before the page is published.


Patch size and stride
---------------------

* **Use 128-pixel patches with a stride of 32 and an apodization border of 24 pixels**, as ``examples/spot_8542``
  now does. Against 64-pixel patches, with the marginal loss: object error 13 to 23 % lower, PSF transfer error
  29 to 45 % lower, also when the seeing changes across the field. The cost is about three times the run time on a
  512-pixel burst.
* With 64-pixel patches the patch edges bias the fit: iterating longer makes the object worse (D/r0 5: error 0.098
  after 10 LBFGS iterations, 0.139 after 30). With 96 to 128 pixels it does not.
* When the seeing changes across the field (anisoplanatism), a stride of 64 is unsafe: with 128-pixel patches the
  mean barely improves (−0.3 %) and 3 of 12 scenes come out more than 25 % worse. A stride of 32 holds.
* Larger patches do not help further under anisoplanatism: with a turbulent layer at 15 km, 160 and 192 pixels give
  −18.9 % and −15.8 % against −20.7 % for 128 pixels.


Object filter: ``scharmer`` or ``garrote``
------------------------------------------

* With 64-pixel patches and the joint loss, ``garrote`` lowers the object error by 8.0 % (29 of 31 scenes), all of
  it above half the diffraction cutoff, and leaves the PSFs unchanged.
* With 128-pixel patches ``scharmer`` (Löfdahl–Scharmer) is better: −23.0 % against −20.8 % for ``garrote``, both
  relative to 64-pixel patches with ``scharmer`` (joint loss). It is also better when the true PSFs are given, so the
  difference is the filter, not the fit. The default stays ``scharmer``.
* With few frames (6) ``garrote`` makes no difference (−0.7 %).
* ``garrote`` changes the object smoothly when the data change slightly; the ``scharmer`` mask switches frequencies
  on or off, and a change of one part in :math:`10^4` in the frames can make the object jump.

.. TODO(check): the 64 px vs 128 px garrote numbers are joint loss; give the marginal-loss numbers if the bench has them.


What cannot be recovered
------------------------

* Above half the diffraction cutoff nothing is recovered, not even with the true PSFs. With 64-pixel patches the
  code adds false power there (three times the true power at D/r0 10).


The marginal loss: when to stop fitting the object spectrum
------------------------------------------------------------

* ``stop_psd_optimization`` freezes the parameters of the object power spectrum after that many modes have been
  added. Freezing early gives PSFs that are too narrow; fitting to the end (as the code did before it honoured the
  option) gives PSFs that are too broad.
* Stopping halfway, at 20 modes (5 of the 10 outer steps of the example), was better than fitting to the end in 11
  of 13 synthetic scenes: object error 0.455 → 0.304, PSF error 0.465 → 0.439 (64-pixel patches, LBFGS 10).

.. TODO(check): repeat the stop_psd_optimization sweep with 128-pixel patches before stating it for the new example.


The joint loss: level of the object prior
-----------------------------------------

* The best ``s_u_joint`` grows linearly with the noise and with the inverse of the number of frames:
  about :math:`250\,(\sigma/0.02)\,(12/N)`. With 64-pixel patches the prior hurts (+57 % at twice the noise).

.. TODO(check, UiO): wording of "the prior hurts with 64-pixel patches" — hurts relative to what?


The optimizer
-------------

* LBFGS amplifies tiny changes: a change of one part in :math:`10^7` in the frames changes the object by 32 to 35 %
  (Adam: by :math:`10^{-6}`). The same configuration run on another machine gives a different result: the PSF error
  of one scene differed by up to 0.2 between a Mac and a Linux CPU, and by 3 to 4 % between two GPU types.
  **Compare configurations only on the same machine**, and over many scenes.


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

* None of the measures that need no truth picks the best configuration better than simply using 128-pixel patches
  (241 scenes). The final loss and the :math:`\chi^2` at high frequencies are misleading: they reward fitting the
  noise. The Fourier ring correlation between two halves of the burst is the best of them, and still not enough.
* For PSFs, the error of each frame's :math:`|{\rm OTF}|` does not predict how good the object will be; the transfer
  error :math:`R = \sum \hat T^* T / \sum |\hat T|^2` does, and is the one ``benchmarks/`` reports.


Tried and not adopted
---------------------

* **Restricting the wavefront to the directions the data determine** (experimental option, off by default; see
  :ref:`configuration`): it helps synthetic scenes (−11 %) but makes the real sunspot worse at D/r0 5 (+9 %, 6 of 6
  scenes, worse PSFs). Leave it off unless your scenes are tested with ``benchmarks/``.
* A power-law object spectrum in the joint loss instead of the constant ``s_u_joint``.
* Weighting each frame by its own noise in the loss.
* A port of the ``scharmer`` filter to the GPU: the same masks, but slower (479 ms against 19 ms on an Apple GPU,
  30 patches) because its flood fill is sequential. ``garrote`` runs on the GPU and is up to 10 times faster than
  ``scharmer`` with many patches.

.. TODO(check, UiO): power-law prior and per-frame noise weighting were scored before the units fix — rerun with
   benchmarks/ or drop them. Confirm the option name of the wavefront restriction and the -11 % / +9 % numbers.
