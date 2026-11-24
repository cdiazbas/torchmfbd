.. include:: ../code_name
.. _configuration:


Mosaicking
==========

When dealing with spatially variant PSFs, the deconvolution is often done via mosaicking. ``torchmfbd`` provides a simple tool
for simplifying the mosaicking and recombination process. The mosaicking is done by instantiating the ``torchmfbd.Patchify4D`` class, which
has two methods, ``patchify`` and ``unpatchify``. The ``patchify`` method takes a tensor of shape ``(n_seq, n_frames, n_x, n_y)`` and
returns another tensor of shape ``(n_seq, n_patches, n_frames, n_x_patch, n_y_patch)``. The ``unpatchify`` method takes a tensor of shape
``(n_seq, n_patches, n_frames, n_x_patch, n_y_patch)`` previously patchified with ``patchify`` and returns a tensor of shape 
``(n_seq, n_frames, n_x, n_y)``. The patchifying is done by splitting the frames into patches of size ``patch_size`` and 
stride ``stride_size``. By default, the patches are flattened along the patches dimension.

::

    import torchmfbd
    patchify = torchmfbd.Patchify4D()
    frames = torch.zeros((4, 12, 256, 256))
    patches = patchify.patchify(frames, patch_size=64, stride_size=64, flatten_sequences=True)
    
In this example, ``patches`` has shape ``(64, 12, 64, 64)``. If we use a different stride, the number of patches will change accordingly:

::
    
    patches = patchify.patchify(frames, patch_size=64, stride_size=32, flatten_sequences=True)

generates a tensor of shape ``(196, 12, 64, 64)``.

If the image moves between the frames by more than a pixel or two, the window of a fixed cutout covers a different piece of
the scene in every frame. MOMFBD cuts each frame of each patch again where it moved (``MAX_LOCAL_SHIFT``, 5 px by
default). Here this is ``images.follow_motion`` in the configuration: the patches are cut with a margin of that many
pixels, and the deconvolution cuts every frame inside it.

::

    config['images']['follow_motion'] = 5
    patches = patchify.patchify(frames, patch_size=128, stride_size=32, flatten_sequences=True, margin=5)
    dec = torchmfbd.Deconvolution(config)
    dec.add_frames(patches, ...)            # patches of 138 px; the fit uses 128
    dec.deconvolve(...)
    obj = patchify.unpatchify(dec.obj, ...)  # 128-px objects, on the grid of the patches

For every frame of every patch the shift is the peak of the cross-correlation of its central cutout with the mean of the
frames aligned with the current shifts (Hann window, three iterations from zero shifts), referred to the median of the
burst. The correlation maps of all the objects added with ``add_frames`` are summed before the peak is taken, so that the
objects that share the wavefront (the cameras of a burst, the channels of phase diversity) get the same shift; they must
be at one wavelength. As in MOMFBD the shift is kept in tip-tilt units and added to the fitted tip-tilt: the tip-tilt
referred to the mean of the frames, and the one compared between neighbouring patches by ``regularization.coupling``, is
the total one, so the object stays where the mean of the uncut frames is. The shifts are in ``dec.motion_shifts``.
It needs all the sequences in one batch (``simultaneous_sequences``), a wavefront PSF model, objects at one wavelength
and no time series; otherwise, or with frames without the margin, the fit goes on with the central cut of the frames
and a warning says why. It is off by default (``images.follow_motion: 0``).

On the 55 benchmark scenes (one object, 128-px patches, stride 32, the example's kl.yaml with the coupling of neighbours),
the change of the object error against the fixed cutout, as a function of the frame motion (rms per axis), on a V100S:

======== ============================== ===============================
motion   ``patchify(follow_motion=5)``   ``images.follow_motion = 5``
======== ============================== ===============================
0 px     +0.2 %                          -0.0 %
3 px     -5.0 %                          -9.4 %
6 px     -18.8 %                         -39.2 % (better in 53 of 55)
======== ============================== ===============================

It costs 4 % of the fit time (2.60 s instead of 2.49 s per scene). The first column is the older pre-alignment in
``patchify``, which registers each patch to the plain mean of its frames, without a window and without the tip-tilt
bookkeeping: the window and the iterations make the difference (-37.9 % at 6 px without the bookkeeping), the bookkeeping
adds 1.6-1.9 %. With two objects that share the wavefront, at 6 px: -35.8 %, against -19.4 % for the pre-alignment with
shared shifts (both with ``follow_motion_refine``, below).

``images.follow_motion_refine`` adds MOMFBD's other step: after the fit, the fitted tip-tilt says where each frame still
is, the frames are cut again there and fitted a second time. On the benchmark it gave nothing over a second pass without
the re-cut (+0.7 % at 0 px, +0.8 % at 3 px, +4.0 % at 6 px) at twice the cost, so it is off by default. A second pass by
itself lowers the error by 3.5 % at 0 px, as more iterations would.

``patchify(follow_motion=...)`` and ``patchify(motion_shifts=...)`` keep the older pre-alignment of the patches, for use
without a deconvolution. The examples use neither: the example's 8542 burst moves by 0.14-0.28 px rms between frames.

The patches can be unpatchified using the ``unpatchify`` method:

::

    frames_back = patchify.unpatchify(patches, apodization=0, weight_type='gaussian', weight_params=1.0)

where an apodization can be applied to each patch, often used to remove border effects during the deconvolution.
Note that the reconstruction of the patches needs to have the same number of dimensions as the original frames, otherwise 
an error will be raised. If you want to reconstruct the frames with a different number of dimensions, remember
to add a singleton dimension to the patches before unpatchifying, for example:

::

    frames_back = patchify.unpatchify(patches[:, None, ...], apodization=0, weight_type='gaussian', weight_params=1.0)

which is useful if, for instance, the patches have shape ``(n_patches, n_x_patch, n_y_patch)``, which is typical
when merging back reconstructed data.

The ``patchify`` function has the following options:

* ``frames``: tensor
    Tensor of shape ``(n_seq, n_frames, n_x, n_y)`` with the frames to be patchified.
* ``patch_size``: int
    Size of the patches. Default is 64.
* ``stride_size``: int
    Stride of the patches. Default is 64.
* ``flatten_sequences``: bool
    If True, the patches are flattened along the patches dimension. Default is True.

The ``unpatchify`` function has the following options:

* ``patches``: tensor
    Tensor of shape ``(n_seq, n_patches, n_frames, n_x_patch, n_y_patch)`` with the patches to be unpatchified.
* ``apodization``: int
    Size of the apodization to be applied to each patch. Default is 0.
* ``weight_type``: str
    Type of apodization to be applied. Default is None. Options are (None, 'cosine', 'gaussian').
* ``weight_params``: float
    Parameter of the apodization. Default is 1.0. The parameter is given in normalized units for 'gaussian' and in pixels for 'cosine'.

The weighting window can always be accessed on the property ``weight`` of the ``Patchify4D`` class. This allows the user to correctly define the weight of the patches to reduce artifacts.