import numpy as np
import torch
import torch.nn.functional as F
import torchmfbd
from torchmfbd.warp import warp, warp_affine
import torchmfbd.util as util
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict

def destretch(frames,             
              ngrid=32, 
              lr=0.01,
              reference_frame=0,
              border=10,
              n_iterations=20,
              lambda_tt=0.1,
              mode='bicubic',
              average_geometry=True,
              n_refine=1
              ):
    
    """
    Perform image destretching on a sequence of frames using gradient-based optimization.
    It optimizes the optical flow in the field-of-view to align the frames to a reference frame.
    To this end, it uses the correlation between the reference frame and the warped frames
    as defined in "Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization"
    by Georgios D. Evangelidis and Emmanouil Z. Psarakis.
    
    Args:
        frames (torch.Tensor): Input tensor of shape (n_seq, n_o, n_f, n_x, n_y) representing the sequence of frames.
        ngrid (int, optional): Grid size for the tip-tilt estimation. Default is 32.
        lr (float, optional): Learning rate for the optimizer. Default is 0.01.
        reference_frame (int or 'avg', optional): Index of the reference frame to which other frames are aligned, or 'avg' for the per-pixel median of the frames. Default is 0.
        border (int, optional): Border size to exclude from the loss computation. Default is 10.
        n_iterations (int, optional): Number of optimization iterations. Default is 20.
        lambda_tt (float, optional): Regularization parameter for the tip-tilt smoothness. Default is 0.1.
        mode (str, optional): Interpolation mode for the warping. Default is 'bicubic' ('nearest'/'bilinear'/'bicubic').
        average_geometry (bool, optional): If True, the estimated flows are centered to have zero mean,
            effectively destretching to the average geometry of the sequence instead of locking
            to the reference frame's distortion. Default is True.
        n_refine (int, optional): Number of refinement steps. In each step, the destretching
            is performed again on the results of the previous step. Default is 1.
    Returns:
        tuple: A tuple containing:
            - warped (torch.Tensor): Warped frames after destretching.
            - tt (torch.Tensor): Estimated tip-tilt values (cumulative across refinements).
    """
    ndim = len(frames.shape)
    if ndim == 3:        
        frames = frames[None, None, ...]

    if ngrid == 1 and lambda_tt > 0.0:
        print('Warning: Regularization is not applied when ngrid=1. Setting lambda_tt to 0.')
        lambda_tt = 0.0


    device = frames.device
    n_seq, n_o, n_f, n_x, n_y = frames.shape

    finite_difference = util.FiniteDifference().to(frames.device)

    # Always re-warp from ORIGINAL frames with the cumulative centered flow.
    # This ensures each refinement step sees frames in average geometry, avoids blur from
    # multiple resamplings, and keeps the centering consistent across steps.
    total_tiptilt = torch.zeros((n_seq, n_f, 2, ngrid, ngrid), device=device)
    im_orig = rearrange(frames, 'nb no nf nx ny -> (nb nf) no nx ny')

    for r in range(n_refine):
        if n_refine > 1:
            print(f"Refinement step {r+1}/{n_refine}")

        # Build current_frames: warp ORIGINAL with the centered cumulative flow so far.
        # Step 0 uses zeros (original frames), subsequent steps use the centered total flow.
        with torch.no_grad():
            tt_so_far = rearrange(total_tiptilt, 'nb nf d nx ny -> (nb nf) d nx ny')
            # Use the requested interpolation mode for the flow field
            tt_so_far = F.interpolate(tt_so_far, size=(n_x, n_y), mode=mode, align_corners=False if mode != 'nearest' else None)
            current_frames = warp(im_orig, tt_so_far, mode=mode)
            current_frames = rearrange(current_frames, '(nb nf) no nx ny -> nb no nf nx ny', nb=n_seq)

        tiptilt = torch.zeros((n_seq, n_f, 2, ngrid, ngrid), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([tiptilt], lr=lr)  # Same lr each step

        # Define the reference frame 'ir'
        if r == 0:
            # First pass: use the user-selected reference_frame
            if reference_frame == 'avg':
                ir, _ = torch.median(frames, dim=2, keepdim=True)
            else:
                ir = current_frames[:, :, reference_frame:reference_frame + 1, :, :]
        else:
            # Refinement passes: use the MEAN of all warped frames as the reference.
            # This is much more stable (higher SNR).
            if n_refine > 1:
                print(f"  --> Using average frame as reference for refinement...")
            ir = torch.mean(current_frames, dim=2, keepdim=True)

        ir = ir - torch.mean(ir, keepdim=True, dim=(-1, -2))

        t = tqdm(range(n_iterations))
        im_refine = rearrange(current_frames, 'nb no nf nx ny -> (nb nf) no nx ny')

        for loop in t:

            optimizer.zero_grad()

            tt_iter = rearrange(tiptilt, 'nb nf d nx ny -> (nb nf) d nx ny')            
            tt_iter = F.interpolate(tt_iter, size=(n_x, n_y), mode='bilinear')
            warped = warp(im_refine, tt_iter, mode='bilinear')
            warped = rearrange(warped, '(nb nf) no nx ny -> nb no nf nx ny', nb=n_seq)

            iw = warped - torch.mean(warped, keepdim=True, dim=(-1, -2))

            t1 = ir / torch.linalg.norm(ir, dim=(-1, -2), keepdim=True)
            t2 = iw / torch.linalg.norm(iw, dim=(-1, -2), keepdim=True)

            if border > 0:
                t1 = t1[..., border:-border, border:-border]
                t2 = t2[..., border:-border, border:-border]

            loss_corr = torch.linalg.norm(t1 - t2, dim=(-1,-2))
            loss_corr = torch.mean(loss_corr)

            regularization = torch.tensor(0.0, device=device)
            if lambda_tt > 0.0:
                tmp = rearrange(tiptilt, 'nb nf nm nx ny -> (nb nf) nm nx ny')
                regularization = lambda_tt * torch.mean(finite_difference(tmp)**2)

            loss = loss_corr + regularization
            loss.backward()
            optimizer.step()

            tmp = OrderedDict()
            tmp['Lcorr'] = f'{loss_corr.item():8.6f}'
            tmp['R'] = f'{regularization.item():8.6f}'
            tmp['L'] = f'{loss.item():8.6f}'
            t.set_postfix(ordered_dict=tmp)

        # Accumulate this step's incremental flow into total_tiptilt.
        # current_frames for the next step is rebuilt at the top of the loop
        # by warping im_orig with the updated total_tiptilt.
        with torch.no_grad():
            total_tiptilt += tiptilt

    # Final output: apply average geometry centering to the cumulative flow and
    # warp the ORIGINAL frames exactly once to avoid any blur accumulation.
    with torch.no_grad():
        if average_geometry:
            total_tiptilt -= torch.mean(total_tiptilt, dim=1, keepdim=True)

        tt_out = rearrange(total_tiptilt, 'nb nf d nx ny -> (nb nf) d nx ny')
        # Use the requested interpolation mode for the final flow field
        tt_out = F.interpolate(tt_out, size=(n_x, n_y), mode=mode, align_corners=False if mode != 'nearest' else None)
        warped = warp(im_orig, tt_out, mode=mode)
        warped = rearrange(warped, '(nb nf) no nx ny -> nb no nf nx ny', nb=n_seq)

        tt_out = rearrange(tt_out, '(nb nf) d nx ny -> nb nf d nx ny', nb=n_seq)

    if ndim == 3:
        warped = warped[0, 0, ...]

    return warped.detach(), tt_out.detach()


def apply_destretch(frames, tt, mode='bicubic'):

    n_seq, n_o, n_f, n_x, n_y = frames.shape
    
    im = rearrange(frames, 'nb no nf nx ny -> (nb nf) no nx ny')
    
    warped = warp(im, tt, mode=mode)

    warped = rearrange(warped, '(nb nf) no nx ny -> nb no nf nx ny', nb=n_seq)

    return warped


def align(frames,              
              lr=0.01,              
              border=0,
              region=None,
              n_iterations=20,              
              mode='bilinear',
              padding_mode='reflection',
              no_shear=False,
              no_rotation=False,
              return_grid=False
              ):
    
    """
    Perform image alignment between two images using gradient-based optimization.
    It optimizes the affine transformation matrix to align the second frame to the first frame, which is used as reference.
    To this end, it uses the correlation between the reference frame and the warped frames
    as defined in "Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization"
    by Georgios D. Evangelidis and Emmanouil Z. Psarakis.
    
    Args:
        frames (torch.Tensor): Input tensor of shape (n_f, n_x, n_y) representing the sequence of frames.        
        lr (float, optional): Learning rate for the optimizer. Default is 0.01.        
        border (int, optional): Border size to exclude from the loss computation. Default is 10.
        n_iterations (int, optional): Number of optimization iterations. Default is 20.        
        mode (str, optional): Interpolation mode for the warping. Default is 'bilinear' ('nearest'/'bilinear').        
        padding_mode (str, optional): Padding mode for the warping. Default is 'reflection' ('zeros'/'border'/'reflection').
        no_shear (bool, optional): If True, only optimize for rotation and translation (no shear). Default is False.
        no_rotation (bool, optional): If True, only optimize for a translation. Default is False.
        return_grid (bool, optional): If True, also return the sampling grid of the final warp. Default is False.
    Returns:
        tuple: A tuple containing:
            - warped (torch.Tensor): Warped frames after alignment.
            - tt (torch.Tensor): Estimated affine matrix.
    """
    
    device = frames.device
    n_f, n_x, n_y = frames.shape

    finite_difference = util.FiniteDifference().to(frames.device)
    
    
    if no_rotation:
        # translation only: the rotation/scale/shear block stays the identity
        pars = torch.zeros(2, dtype=torch.float32, device=device, requires_grad=True)
        affine = torch.eye(2, 3, dtype=torch.float32, device=device)[None]
        optimizer = torch.optim.Adam([pars], lr=lr)
    elif no_shear:
        pars = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
        
        affine = np.zeros((1, 2, 3), dtype=np.float32)
        affine[:, 0, 0] = 1.0
        affine[:, 1, 1] = 1.0
        affine = torch.tensor(affine, device=device, requires_grad=True)        
        optimizer = torch.optim.Adam([pars], lr=lr)
    else:        
        affine = np.zeros((1, 2, 3), dtype=np.float32)
        affine[:, 0, 0] = 1.0
        affine[:, 1, 1] = 1.0
        affine = torch.tensor(affine, device=device, requires_grad=True)        
        optimizer = torch.optim.Adam([affine], lr=lr)

    # Reference frame
    ir = frames[0:1, :, :]
    im = frames[1:2, :, :]

    ir = ir - torch.mean(ir, keepdim=True, dim=(-1, -2))
    
    t = tqdm(range(n_iterations))
    
    for loop in t:

        optimizer.zero_grad()

        if no_rotation:
            affine = torch.eye(2, 3, dtype=torch.float32, device=device)[None].clone()
            affine[:, 0, 2] = pars[0]
            affine[:, 1, 2] = pars[1]
        elif no_shear:
            affine = torch.zeros((1, 2, 3), dtype=torch.float32, device=device)
            affine[:, 0, 0] = torch.cos(pars[0])
            affine[:, 0, 1] = -torch.sin(pars[0])
            affine[:, 1, 0] = torch.sin(pars[0])
            affine[:, 1, 1] = torch.cos(pars[0])
            affine[:, 0, 2] = pars[1]
            affine[:, 1, 2] = pars[2]
                
        warped = warp_affine(im[None, ...], affine, mode='bilinear', padding_mode=padding_mode)[0, ...]
        
        iw = warped - torch.mean(warped, keepdim=True, dim=(-1, -2))

        t1 = ir / torch.linalg.norm(ir, dim=(-1, -2), keepdim=True)
        t2 = iw / torch.linalg.norm(iw, dim=(-1, -2), keepdim=True)

        if border > 0:
            t1 = t1[..., border:-border, border:-border]
            t2 = t2[..., border:-border, border:-border]

        if region is not None:
            x1, x2, y1, y2 = region            
            t1 = t1[..., x1:x2, y1:y2]
            t2 = t2[..., x1:x2, y1:y2]

        loss = torch.linalg.norm(t1 - t2, dim=(-1,-2))
        loss = torch.mean(loss)
                        
        loss.backward()

        optimizer.step()

        tmp = OrderedDict()        
        tmp['L'] = f'{loss.item():8.6f}'
        
        t.set_postfix(ordered_dict = tmp)

    out = warp_affine(im[None, ...], affine, mode=mode, padding_mode=padding_mode, return_grid=return_grid)
        
    if return_grid:        
        warped, grid = out
        return warped[0, ...].detach(), affine.detach(), grid.detach()
    else:
        return out[0, ...].detach(), affine.detach()


def apply_align(frames, affine, mode='bicubic', padding_mode='reflection'):
        
    warped = warp_affine(frames, affine, mode=mode, padding_mode=padding_mode)

    return warped

def align_sequence(frames, 
                   lr=0.01,
                   border=0,
                   region=None,
                   n_iterations=20,
                   mode='bilinear',
                   padding_mode='reflection',
                   no_shear=False,
                   no_rotation=False):
    
    frames_out = frames.clone()

    avg = torch.mean(frames, dim=0, keepdim=True)

    nframes = frames.shape[0]

    frames_out[0, ...] = frames[0, ...]    

    for i in tqdm(range(nframes)):

        inp = torch.cat([avg, frames[i:i+1, :, :]], dim=0)

        out, affine = torchmfbd.align(inp,
                lr=lr,
                border=border,
                region=region,
                n_iterations=n_iterations,              
                mode=mode,
                padding_mode=padding_mode,
                no_shear=no_shear,
                no_rotation=no_rotation
                )
    
        frames_out[i, ...] = out[0, :, :]

    return frames_out
        # affine_out[i, ...] = affine.cpu().numpy()