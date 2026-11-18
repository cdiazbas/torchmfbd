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


The example as it ships: joint loss, ``s_u_joint: auto`` and patch coupling
---------------------------------------------------------------------------

``examples/spot_8542`` uses the joint loss with the object prior level taken from the noise (``s_u_joint: auto``) and
the ``coupling`` regularization (``lambda`` 30, ``n_modes`` 10), which pulls together the low-order wavefront
coefficients of neighbouring patches. Against the marginal loss it used before (128-pixel patches, stride 32,
LBFGS 10; each number is the mean of the per-scene changes):

* 42 scenes with neighbouring patches (real sunspot crops and two-layer seeing), the example as shipped: object
  error 11.1 % lower, 33 better (sign test p = 0.0003), none more than 25 % worse; with seeing that changes across
  the field, real crops −20.0 %;
* 55 scenes with 6 frames: −10.6 %, 43 better, none more than 25 % worse;
* 13 synthetic one-patch scenes: −20.3 % (13 of 13).

The last two were measured with the same three settings given on the command line of ``benchmarks/run.py``, before
the example's configuration was changed.

Confirmed on hold-out scenes, which were not used to choose the configuration (every seed changed; 128-pixel
patches, stride 32, Tesla V100S): −15.4 % on 42 scenes with neighbours (35 better, none more than 25 % worse), −11.9 %
on 55 scenes with 6 frames (38 better) and −24.8 % on 13 synthetic scenes (13 better). Two starts of the configuration
differ by 1.0 to 2.1 % per scene on these sets. On a Tesla T4 (another GPU type) the hold-out gain on the 42 scenes
is −12.9 % (35 better; one real scene more than 25 % worse).

All three parts are needed:

* The coupling with the marginal loss gives nothing measurable (−1.4 %, inside its ±8 % spread).
* The joint loss with the coupling but a fixed prior level fails on 6-frame bursts (6 real scenes more than 25 %
  worse); ``s_u_joint: auto`` prevents it.
* The coupling needs all patches of an image in one batch (``simultaneous_sequences`` at least the number of
  patches) and the pairs of neighbours (``set_neighbours(patchify.neighbours())``), as the example scripts do.
* The coupling does not rescue a stride of 64: with the marginal loss it is 23.5 % worse than a stride of 32, and a
  coupling of 100 does not change that.


Object filter: ``scharmer`` or ``garrote``
------------------------------------------

Measured with the joint loss only.

* With 64-pixel patches, ``garrote`` lowers the object error by 8.0 % (29 of 31 scenes), mostly above half the
  diffraction cutoff, and leaves the PSFs unchanged.
* With 128-pixel patches (stride 64) ``scharmer`` (Löfdahl–Scharmer) is better: −23.0 % against −20.8 % for
  ``garrote``, both relative to 64-pixel patches with ``scharmer``. It is also better when the true PSFs are given
  (0.121 against 0.124), so the difference is the filter, not the fit. The default stays ``scharmer``.
* With 6 frames and 64-pixel patches ``garrote`` makes no difference (−0.7 %).
* On top of the example's coupling (128-pixel patches) ``garrote`` is slightly worse (+1.5 %).
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

The example now uses the joint loss; this applies when the marginal loss is chosen.

* ``stop_psd_optimization`` freezes the parameters of the object power spectrum after that many modes have been
  added. Freezing early gives PSFs that are too narrow; fitting to the end (as the code did before it honoured the
  option) gives PSFs that are too broad.
* With 128-pixel patches (stride 32, border 24), any early stop beats fitting to the end.
  At the shipped value of 20 the object error drops from 0.315 to 0.251 and the PSF transfer error from 0.274 to
  0.226 (12 of 13 synthetic scenes better); with the seeing changing across the field, 0.338 → 0.263 (12 of 12).
  Stopping at 5 or 14 gives about the same (0.230 and 0.233 on the synthetic set): the differences between the three
  are within the run-to-run spread of these sets (±8.6 %), so there is no reason to move from 20.
* With 64-pixel patches the result was the same: stopping at 20 was better than fitting to the end in 11 of 13
  synthetic scenes (object error 0.455 → 0.304, PSF error 0.465 → 0.439).


The joint loss: level of the object prior
-----------------------------------------

* With 96- and 128-pixel patches the best ``s_u_joint`` grows linearly with the noise and with the inverse of the
  number of frames: about :math:`250\,(\sigma/0.02)\,(12/N)` for frames divided by their mean (measured at 96
  pixels, checked at 128). ``s_u_joint: auto`` applies this rule.
* With 64-pixel patches do not apply that rule: it makes the object worse than the fixed ``s_u_joint = 250``
  (+6.5 % in standard conditions, +9.2 % with 6 frames, +57 % with twice the noise; 31 scenes).


The optimizer
-------------

* LBFGS amplifies tiny changes: a change of one part in :math:`10^7` in the frames changes the object by 32 to 35 %
  (Adam: by :math:`10^{-6}`). The same configuration run on another machine gives a different result: the PSF error
  of one scene differed by up to 0.24 between a Mac and a Linux CPU, and by 3 to 4 % between two GPU types (V100S and
  T4). **Compare configurations only on the same machine**, and over many scenes. Two starts of the same
  configuration differ by about ±8 % per scene with the marginal loss and ±2.5 % with the joint loss: a difference
  smaller than that, scene by scene, is not a result.


Time series (``filter_loss: wiener_time``)
------------------------------------------

* Each time step is weighted by its own noise. A step ten times noisier than the rest no longer spoils the PSFs of
  the whole series (error of the quiet steps 1.17 → 0.46, against 0.45 with no noisy step); with twice the noise the
  effect is small (0.50 → 0.47) and with 1.3 times it is none.
* ``regularization.time.mask_window`` (default 3) computes the ``scharmer`` mask, or the ``garrote`` gain, of each
  step from the step and its neighbours. With 1, a badly degraded step keeps an error of 0.72; with 3 it comes back to
  the error of a clean step (0.19).
* With seeing that changes across the field and an evolving scene (6 synthetic scenes of 4 steps, 128-pixel
  patches, the example's joint loss), the time term is the large gain: −31.8 % at a weight of 1.5e-4 (6 of 6). The
  coupling adds about 4 % on top at that weight (−3.9 %, 5 of 6, with the second layer at 5 km; −3.8 %, 6 of 6, at
  15 km), and nothing without the time term or with a weight of 5e-4, which is itself 1.5 % worse than 1.5e-4. The
  coupling acts within each time step, never across steps.
* With the best weight of the time term the first and last steps improve less than the middle ones, because they have
  one neighbour. Penalizing the curvature of the time evolution instead of the difference between steps does not
  change this (mean error 0.224 → 0.219) and makes a series with one bad step worse (0.202 → 0.214).

The real Sun in time
^^^^^^^^^^^^^^^^^^^^

The synthetic evolution above is a mix of two random fields. The ``real-time`` set of ``benchmarks/`` follows the real
solar evolution of the SST/CRISP 8542 sequence of ``examples/spot_8542_time``: the sharpest frame of each step is the
truth, and 3 windows of 4 steps are degraded with two-layer seeing at D/r0 5 and 10 (6 scenes, 12 frames, noise 0.02).
The truth keeps the sequence's own noise (σ/μ ≈ 0.015), about 0.09 of its contrast inside the cutoff, which no
configuration can go below.

* **The weight of the time term.** The weight that suits the synthetic scenes (1.5e-4) is too weak for the real Sun:
  on hold-out scenes at D/r0 10 it is worse than no time term (0.396 against 0.367), and 5e-4 is better than both
  (0.348; mean −5.8 % against 1.5e-4). At D/r0 5 both weights bring 0.336 to 0.202. Between 5e-4 and 1.5e-3 the result
  is flat; on the synthetic set 5e-4 costs +5.3 % on the default scenes and nothing on the hold-out ones. Without the
  time term at D/r0 10 the object is sharp but false; with too weak a weight it stays almost as blurred as a frame.
* **Bad steps already borrow from good ones.** With the seeing alternating between D/r0 5 and 10 from step to step,
  the time term brings the bad steps from 0.37 to 0.24, next to the good ones (0.22). Equalising the power spectrum
  of each step towards the sharpest one after the fit adds nothing (+0.2 %).
* **How the Sun changes between steps.** Over one step the fraction of the power that changes grows from 1.5 % at the
  largest scales to 100 % at 0.6 of the cutoff. Yet taking time weight away where the Sun renews is worse (+1.6 to
  +6.6 %): at those frequencies the term averages the noise of neighbouring steps, which is worth more than the
  evolution it blurs. Giving them more weight is not robust either (−2.0 %, then −0.4 % on hold-out scenes).
* **Solar motion.** The Sun moves 0.3 px per step (median); removing the motion perfectly would improve the object by
  2 to 6 %. A second fit on frames aligned with the flow of the first fit's objects gains about 2 % (−2.8 and −3.8 %
  on hold-out scenes, 6 of 6, but +0.4 % on the default ones of one GPU) at twice the time; the flow of the mean
  frames is worse (+3.5 %).
* **Not useful here:** a window of 8 steps instead of two of 4 (+2.7 %), stride 16 (−0.3 % at 3 times the time),
  more iterations (+19 %), 96-pixel patches (+16 %), no coupling (+5 %) or coupling 100 (+3 %), ``mask_window`` 1 or
  5 (±2 %), and a mean high-order halo in the OTF (+4.7 %, even with the true D/r0). ``s_u_joint`` has no effect
  with ``wiener_time``.
* ``wiener_time_contrast`` weighted the time term by a contrast divided by the mean of apodized patches, which is
  near zero: +63 % on these scenes before the fix. Even computed right it only weakens the time term on the better
  steps, which is worse at D/r0 10.


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
  +0.88 on synthetic scenes), and is the one ``benchmarks/`` reports. When the marginal fit runs to the end, for
  instance, the per-frame error is at its lowest while the object and the transfer error are at their worst.


Tried and not adopted
---------------------

* **Restricting the wavefront to the directions the data determine** (``psf.nullspace_d_r0``, with
  ``psf.nullspace_threshold``; experimental, off by default, see :ref:`configuration`): it helps synthetic scenes
  (mean error 0.265 → 0.235 over 13 scenes, −11 %) but makes the real sunspot worse at D/r0 5 (+9 %, 6 of 6 scenes),
  with worse PSFs in 3 of 4 groups (64-pixel patches, joint loss). With 128-pixel patches it is worse still on the
  real scenes (+14 %). Leave it off unless your data are tested with ``benchmarks/``.
* A power-law object spectrum in the joint loss instead of the constant ``s_u_joint``: at best it ties the constant
  (0.268 against 0.267, better in 7 of 13 synthetic scenes); every other variant is worse (0.29 to 0.45) and gives
  worse PSFs.
* Weighting each frame by its own noise in the loss: the object gets worse (0.604 → 0.759 with Adam, 0.586 → 0.718
  with LBFGS; 32-pixel scenes with a different noise per frame, 5 scenes).
* A port of the ``scharmer`` filter to the GPU: the same masks, but slower (479 ms against 19 ms on an Apple GPU,
  30 patches) because its flood fill is sequential. ``garrote`` runs on the GPU and is up to 10 times faster than
  ``scharmer`` with many patches.


Around the example as it ships: what does not improve it
--------------------------------------------------------

Measured on 24 scenes chosen to be hard (real crops at D/r0 5 and 10, and anisoplanatic real crops), 128-pixel
patches, the example's configuration as the reference; each number is the mean of the per-scene changes. A second
start perturbed by :math:`10^{-3}` rad moves these scenes by ±7.3 % on average, so smaller changes are not results.

* **Stride.** 64 instead of 32 halves the time but loses with anisoplanatism (+5 to +11 %). Use a stride that tiles
  the field: with 48 on 256-pixel scenes the last rows are not covered by any patch.
* **Coupling.** ``lambda`` 10, 100 or 300 and coupling 5 or 20 modes instead of 10: no gain (−0.0 to +5.9 %). 96-pixel
  patches with a stronger coupling (30, 100, 300) are worse (+8 to +11 %) and their PSFs much worse (transfer error
  +26 to +29 %).
* **Longer fits** (15 or 20 outer iterations, ``lbfgs_max_iter`` 30), **a stronger object prior** (``s_u_joint`` auto
  times 2) and, the other way round, a weaker one (times 0.5, +12 %): the first two help the isoplanatic real crops
  (−3 to −16 %) and hurt the anisoplanatic ones (+4 to +25 %). With the PSF varying inside the patch, fitting more
  closely fits that variation instead of the object.
* **Fewer iterations, for speed**: 8 outer iterations (+3.8 %), ``lbfgs_max_iter`` 15 (+3.5 %); 6 iterations or
  ``lbfgs_max_iter`` 10 are much worse (+13 to +14 %). Intermediate prior levels (auto times 0.75 or 1.5) and a
  higher object cutoff (0.98/0.995) do not help either.
* **Mode annealing.** All modes from the start, or ending at 30 % or 90 % of the iterations, or a sigmoid schedule:
  worse (+3.5 to +16 %) than the linear 10–60 % of the example.
* **Loss filter** ``lowpass``: +2.6 % and 2.8 times slower. **Object cutoff** 0.85/0.90 or 0.90/0.95: no gain.
* **Two starts** (the usual one and a perturbed one), keeping the better whole fit or the better start per patch:
  no gain (−0.1 / +0.2 %), 1.5 times slower. The example's zero start had no failed scene; the failures of the
  perturbed start were caught, but there was nothing to catch in the zero start.
* **Down-weighting frames with a large residual** (iterative reweighting): +1.8 %, worse in 18 of 24 scenes.
* **A Kolmogorov prior on the wavefront coefficients**: +8.6 %, worse in 19 of 24.

With the true PSFs the object would be 30 to 48 % better on the real and anisoplanatic scenes (13 % at D/r0 3 and
6 to 8 % on the synthetic ones): the PSF estimate is what limits the result, not the object filter.

Where the time goes: on a T4, a fit of 25 patches of 128 pixels with 12 frames takes 6.8 s, of which evaluating the
PSFs and the loss is a third, in about 25 evaluations per outer iteration; the rest is mostly the gradients and the
line search of LBFGS. Faster fits need fewer evaluations, not a cheaper loss.

What limits the result: the PSF inside a patch
----------------------------------------------

With less noise the example's object gets *worse* on scenes whose seeing changes across the field, while the object
from the true PSFs gets better (24 hard scenes; anisoplanatic real crops at D/r0 5: 0.129 → 0.204 when the noise
goes from 0.02 to 0.0005, against 0.067 → 0.042 with the true PSFs). Twice the frames barely help (−0 to −14 %). The
limit is the model, one PSF per patch, not the noise or the number of frames: the cleaner the data, the more the fit
follows the variation of the PSF inside the patch.

* A stronger object prior at low noise does not help (+3.6 to +8 %).
* The data term of the joint loss is divided by the noise variance and the coupling is not, so at low noise the
  coupling is relatively weaker. Scaling ``lambda`` up helps on the screen scenes below a noise of about 0.005
  (−3 to −5 %), not at 0.01 or 0.04, and on hold-out scenes the gain shrinks to −1.5 to −2.4 %, inside the noise
  floor. ``lambda`` 30 is kept at every noise level.
* ``DeconvolutionSV`` (a PSF that varies across the field) with ``examples/spot_sv_8542/nmf_sv.yaml`` as it is: 18 to
  38 % worse than the example on the anisoplanatic real crops, and 4 to 5 times slower at 150 iterations. It is not
  tuned (its own loss, Adam, the object as a free variable); this is a starting point, not a verdict.
