import numpy as np
import torch.nn.functional as F
import torch
import warnings
from einops import rearrange

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class Patchify4D(object):
    def __init__(self):
        super().__init__()

    def _follow_motion(self, x, p, max_shift):
        """
        Each frame of each patch cut again where it moved (the trick of MOMFBD/redux, which moves the cutout of every
        frame by its whole-pixel tilt). The shift of a frame is the peak of its cross-correlation with the mean of the
        frames of the patch (means removed), limited to +-max_shift; the frame is then cut from the frames padded by
        reflection, so that the window covers the same piece of the scene in every frame.

        On the 55 benchmark scenes (128-px patches, stride 32, V100S), against the fixed cutout, with max_shift 5 / 10:
        frame motion 0 px +0.2 / +1.3 %, 1.5 px -0.2 / -0.4 %, 3 px -5.0 / -5.2 %, 4.5 px -14.1 / -14.3 %, 6 px
        -18.8 / -20.1 % (52/55); at 6 px also -21.8 % on the hold-out scenes. 5 is MOMFBD's default MAX_LOCAL_SHIFT
        (the examples do not use it: their 8542 burst moves 0.14-0.28 px rms between frames).

        Args:
            x (torch.Tensor): frames (n_batch, n_frames, nx, ny).
            p (torch.Tensor): their patches (n_batch, n_patches, n_frames, K, K), in unfold order.
            max_shift (int): largest shift in pixels.
        Returns:
            torch.Tensor: the patches cut again, same shape as p.
        """
        B, L, Fr, K, _ = p.shape
        m = max_shift
        dev = p.device
        q = p - p.mean(dim=(-1, -2), keepdim=True)
        ref = q.mean(dim=2, keepdim=True)
        # Peak of ifft(F(ref) F(frame)*): the shift that registers the frame to the mean
        r = torch.fft.ifft2(torch.fft.fft2(ref) * torch.conj(torch.fft.fft2(q))).real
        idx = torch.argmax(r.reshape(B, L, Fr, K * K), dim=-1)
        sy, sx = idx // K, idx % K
        sy = torch.where(sy > K // 2, sy - K, sy)
        sx = torch.where(sx > K // 2, sx - K, sx)
        dy, dx = (-sy).clamp(-m, m), (-sx).clamp(-m, m)
        self.motion_shifts = torch.stack([dy, dx], dim=-1)
        return self._cut(x, dy, dx, m)

    def _cut(self, x, dy, dx, m):
        """The K x K patches of the frames x (n_batch, n_frames, nx, ny) cut at the grid position plus (dy, dx), each of
        shape (n_batch, n_patches, n_frames), from the frames padded by reflection by m >= max |shift| pixels."""
        B, Fr = x.shape[0], x.shape[1]
        L, K = dy.shape[1], self.K
        dev = x.device
        xp = F.pad(x, (m, m, m, m), mode='reflect')
        li = torch.arange(L, device=dev)
        y0, x0 = (li // self.Ly) * self.S, (li % self.Ly) * self.S
        ar = torch.arange(K, device=dev)
        rows = (y0[None, :, None] + dy + m)[..., None, None] + ar[:, None]
        cols = (x0[None, :, None] + dx + m)[..., None, None] + ar[None, :]
        bi = torch.arange(B, device=dev)[:, None, None, None, None]
        fi = torch.arange(Fr, device=dev)[None, None, :, None, None]
        return xp[bi, fi, rows, cols]

    def patchify(self, x, patch_size=64, stride_size=64, flatten_sequences=True, return_coordinates=False, follow_motion=0,
                 motion_shifts=None, margin=0):
        """
        Splits the input tensor into patches.
        Args:
            x (torch.Tensor): Input tensor of shape (*, n_frames, nx, ny).
            follow_motion (int): 0 (the default) cuts every frame at the same place. Above 0, each frame of each patch is
                cut again where that frame moved: at the whole-pixel shift, at most follow_motion pixels, that registers
                it to the mean of the frames of the patch (see _follow_motion). The patch grid, and so unpatchify, do not
                change; the shifts are kept in self.motion_shifts.
            margin (int): 0 (the default) or the margin, in pixels, of every patch: the patches are then
                patch_size + 2 margin pixels across, on the same grid (the frames are padded by reflection at the edges of
                the field), and unpatchify still takes patch_size patches. For images.follow_motion = margin, where the
                deconvolution cuts every frame again where it moved.
            motion_shifts (torch.Tensor, optional): shifts (n_batch, n_patches, n_frames, 2) to cut with, instead of
                estimating them: the motion_shifts of a previous call. Objects that share the wavefront (the cameras of
                a burst, the channels of phase diversity) must be cut with the SAME shifts, or no common tip-tilt fits
                them: estimate on one and pass its motion_shifts to the others. On the example's 8542 burst the two
                cameras, estimated separately, disagree on 9.5 % of the (patch, frame) shifts.
        """
        self.batch_shape = x.shape[:-3]
        self.n_frames, self.nx, self.ny = x.shape[-3:]
        self.n_batch = int(np.prod(self.batch_shape)) if self.batch_shape else 1

        self.output_size = (self.nx, self.ny)
        self.flatten_sequences = flatten_sequences

        self.P = 0
        self.K = patch_size
        self.S = stride_size
        self.X = self.nx
        self.Y = self.ny
        self.D = 1
        
        self.Lx = np.floor((self.X + 2*self.P - self.D*(self.K-1) - 1) / self.S + 1).astype(int)
        self.Ly = np.floor((self.Y + 2*self.P - self.D*(self.K-1) - 1) / self.S + 1).astype(int)

        # Pixels beyond the last full patch are not covered and do not come back from unpatchify
        rx = self.X - ((self.Lx - 1) * self.S + self.K)
        ry = self.Y - ((self.Ly - 1) * self.S + self.K)
        if rx > 0 or ry > 0:
            warnings.warn(f"patchify: (n - patch_size) is not a multiple of stride_size; the last {rx} rows and "
                          f"{ry} columns are not covered by any patch and unpatchify will not return them")
        
        # We track the mask strictly for one frame internally to make broadcasting/slicing trivial later
        mask_base = torch.ones((*self.batch_shape, 1, self.nx, self.ny), device=x.device)
        mask_flat = mask_base.view(self.n_batch, 1, self.nx, self.ny)
        mask_unfold = F.unfold(mask_flat, kernel_size=self.K, stride=self.S)
                
        x_flat = x.view(self.n_batch, self.n_frames, self.nx, self.ny)
        x_unfold = F.unfold(x_flat, kernel_size=self.K, stride=self.S)
        self.n_patches = x_unfold.shape[-1]

        self.motion_shifts = None
        Km = self.K                                  # size of the patches returned (K, or K + 2 margin)
        if margin > 0:
            if follow_motion > 0 or motion_shifts is not None:
                raise ValueError("margin is for images.follow_motion, which cuts the frames itself: use it without follow_motion")
            xp = F.pad(x_flat, (margin, margin, margin, margin), mode='reflect')
            x_unfold = F.unfold(xp, kernel_size=self.K + 2 * margin, stride=self.S)
            Km = self.K + 2 * margin
        if follow_motion > 0 or motion_shifts is not None:
            p = rearrange(x_unfold, 'b (f x y) L -> b L f x y', x=self.K, y=self.K)
            if motion_shifts is not None:
                ms = torch.as_tensor(motion_shifts, device=x_flat.device).long()
                if tuple(ms.shape) != (p.shape[0], p.shape[1], p.shape[2], 2):
                    raise ValueError(f"motion_shifts has shape {tuple(ms.shape)}, the patches need {(p.shape[0], p.shape[1], p.shape[2], 2)}")
                p = self._cut(x_flat, ms[..., 0], ms[..., 1], max(int(ms.abs().max()), 1))
                self.motion_shifts = ms
            else:
                p = self._follow_motion(x_flat, p, int(follow_motion))
            x_unfold = rearrange(p, 'b L f x y -> b (f x y) L')
        
        if self.flatten_sequences:            
            a = rearrange(x_unfold, 'b (f x y) L -> (b L) f x y', x=Km, y=Km)
            self.mask = rearrange(mask_unfold, 'b (f x y) L -> (b L) f x y', x=self.K, y=self.K)
        else:
            a = rearrange(x_unfold, 'b (f x y) L -> b L f x y', x=Km, y=Km)
            self.mask = rearrange(mask_unfold, 'b (f x y) L -> b L f x y', x=self.K, y=self.K)
            
            if len(self.batch_shape) > 1:
                a = a.view(*self.batch_shape, self.n_patches, self.n_frames, Km, Km)
                self.mask = self.mask.view(*self.batch_shape, self.n_patches, 1, self.K, self.K)

        if return_coordinates:
            x_coord = 1.0 * torch.arange(self.nx, device=x.device) - self.nx // 2
            y_coord = 1.0 * torch.arange(self.ny, device=x.device) - self.ny // 2
            X, Y = torch.meshgrid(x_coord, y_coord, indexing='ij')
            coordX = F.unfold(X[None, None, :, :], kernel_size=self.K, stride=self.S)            
            coordY = F.unfold(Y[None, None, :, :], kernel_size=self.K, stride=self.S)

            if self.flatten_sequences:            
                coordX = rearrange(coordX, '1 (f x y) L -> L f x y', x=self.K, y=self.K).repeat(self.n_batch, 1, 1, 1)
                coordY = rearrange(coordY, '1 (f x y) L -> L f x y', x=self.K, y=self.K).repeat(self.n_batch, 1, 1, 1)
            else:
                coordX = rearrange(coordX, '1 (f x y) L -> 1 L f x y', x=self.K, y=self.K)
                coordY = rearrange(coordY, '1 (f x y) L -> 1 L f x y', x=self.K, y=self.K)
                if len(self.batch_shape) > 1:
                    coordX = coordX.expand(*self.batch_shape, self.n_patches, 1, self.K, self.K)
                    coordY = coordY.expand(*self.batch_shape, self.n_patches, 1, self.K, self.K)

            patchX = torch.mean(coordX, dim=(-1, -2))
            patchY = torch.mean(coordY, dim=(-1, -2))
            return a, torch.cat([patchX, patchY], dim=-1)
        else:
            return a

    def neighbours(self):
        """
        Pairs of adjacent patches of the last patchify call, as indices of the flattened sequences it returned (one
        stride apart in x or in y, within the same image). Pass them to Deconvolution.set_neighbours to couple the
        wavefronts of neighbouring patches (regularization `coupling`). Requires flatten_sequences=True.
        """
        if not self.flatten_sequences:
            raise ValueError("neighbours() needs the patches of a patchify call with flatten_sequences=True")
        pairs = []
        for b in range(self.n_batch):
            base = b * self.n_patches
            for r in range(self.Lx):
                for c in range(self.Ly):
                    p = base + r * self.Ly + c
                    if c + 1 < self.Ly:
                        pairs.append((p, p + 1))
                    if r + 1 < self.Lx:
                        pairs.append((p, p + self.Ly))
        return pairs

    def unpatchify(self, x, apodization=0, weight_type=None, weight_params=None):
        """
        Reconstructs the original image from patches. Works seamlessly if x has
        the original n_frames or if n_frames=1. Patches without a frame axis (one image per
        patch, such as a deconvolved object) give a mosaic without a frame axis.
        """
        xndim = x.ndim

        # One image per patch, with no frame axis: the output has none either
        no_frame_axis = (self.flatten_sequences and xndim == 3) or \
                        (not self.flatten_sequences and xndim == len(self.batch_shape) + 3)

        # Nested layout with the frame axis dropped, (*, L, K, K): restore it
        if not self.flatten_sequences and xndim == len(self.batch_shape) + 3:
            x = x.unsqueeze(-3)
            xndim += 1

        # A stride larger than the trimmed patch leaves pixels that no patch covers
        if self.S > self.K - 2*apodization:
            raise ValueError(f"stride_size = {self.S} is larger than patch_size - 2*apodization = {self.K - 2*apodization}: "
                             f"the mosaic would have uncovered gaps")
        
        # 1. Determine incoming shape properties
        if self.flatten_sequences and xndim == 3:
            x = x[:, None, :, :]  # Add a channel dimension for single-frame input

        if self.flatten_sequences or xndim == 4:
            # Shape is ((prod(*) * L), current_frames, x, y)
            current_frames = x.shape[1]
            x2 = x.clone()
            
            # Extract internal mask (which was saved as shape ((prod(*) * L), 1, K, K))
            mask = self.mask.clone().to(x.device)
        else:
            # Shape is (*, L, current_frames, x, y)
            current_frames = x.shape[-3]
            x2 = x.view(self.n_batch, self.n_patches, current_frames, self.K, self.K)
            x2 = rearrange(x2, 'b L f x y -> (b L) f x y')
            
            mask = self.mask.view(self.n_batch, self.n_patches, 1, self.K, self.K)
            mask = rearrange(mask, 'b L f x y -> (b L) f x y').clone().to(x.device)

        if apodization > 0:            
            x2 = x2[..., apodization:-apodization, apodization:-apodization]
            mask = mask[..., apodization:-apodization, apodization:-apodization]
            
        # 2. Build / Apply Weights
        npix = x2.shape[-2]
        if weight_type is None:
            self.weight = torch.ones((npix, npix), device=x.device)
        elif weight_type == 'gaussian':
            xx = torch.linspace(-1, 1, npix, device=x.device)
            yy = torch.linspace(-1, 1, npix, device=x.device)
            X, Y = torch.meshgrid(xx, yy, indexing='ij')
            R = torch.sqrt(X**2 + Y**2)
            self.weight = torch.exp(-R**2 / weight_params)
        elif weight_type == 'cosine':
            npix_apod = min(weight_params, npix//2)
            # Hann taper without its zero end points: a pixel covered by a single patch (mosaic
            # border, seams when stride = patch size) keeps a non-zero weight
            win = np.hanning(2*npix_apod + 2)[1:-1]
            winOut = np.ones(npix)
            winOut[0:npix_apod] = win[0:npix_apod]
            winOut[-npix_apod:] = win[-npix_apod:]
            weight = np.outer(winOut, winOut)
            self.weight = torch.tensor(weight.astype('float32'), device=x.device)
        
        K = self.K - 2*apodization
        new_size_x = int((self.Lx - 1) * self.S + 1 - 2*self.P + self.D*(K-1))
        new_size_y = int((self.Ly - 1) * self.S + 1 - 2*self.P + self.D*(K-1))        
        output_size = (new_size_x, new_size_y)
        
        x2 = x2 * self.weight[None, None, :, :]
        mask = mask * self.weight[None, None, :, :]                
        
        # 3. Handle the frame-dimension mismatch in F.fold
        # We expand the mask's channel dimension to match x2's current_frames channel size
        mask = mask.expand(-1, current_frames, -1, -1)
        
        x2 = rearrange(x2, '(b L) f x y -> b (f x y) L', b=self.n_batch, L=self.n_patches)
        mask = rearrange(mask, '(b L) f x y -> b (f x y) L', b=self.n_batch, L=self.n_patches)
                
        x2 = F.fold(x2, output_size=output_size, kernel_size=K, stride=self.S)
        mask = F.fold(mask, output_size=output_size, kernel_size=K, stride=self.S)

        final = x2 / (mask + 1e-10)
        
        # 4. View final output respecting the new dynamic current_frames 
        final = final.view(*self.batch_shape, current_frames, new_size_x, new_size_y)

        if no_frame_axis:
            final = final.squeeze(-3)
                
        return final
# class Patchify4D(object):
#     def __init__(self):
#         super().__init__()

#     def patchify(self, x, patch_size=64, stride_size=64, flatten_sequences=True, return_coordinates=False):
#         """
#         Splits the input tensor into patches.
#         Args:
#             x (torch.Tensor): Input tensor of shape (n_scans, n_frames, nx, ny).
#             patch_size (int, optional): Size of each patch. Default is 64.
#             stride_size (int, optional): Stride size for patch extraction. Default is 64.
#             flatten_sequences (bool, optional): If True, the output tensor will have shape (n_scans * n_frames, patch_size, patch_size). Default is True.
#             return_coordinates (bool, optional): If True, the function will return the coordinates of the patches. Default is False.
#         Returns:
#             torch.Tensor: Tensor containing the patches with shape (n_scans, L, n_frames, patch_size, patch_size), where L is the number of patches extracted.            
#         """
                            
#         self.n_scans, self.n_frames, self.nx, self.ny = x.shape

#         self.output_size = (self.nx, self.ny)
#         self.flatten_sequences = flatten_sequences

#         # padding, kernel size, stride, dilation and size of the input
#         self.P = 0.0
#         self.K = patch_size
#         self.S = stride_size
#         self.X = self.nx
#         self.Y = self.ny
#         self.D = 1.0
        
#         # Compute the size of blocks that will be extracted using unfold
#         self.Lx = np.floor((self.X + 2*self.P - self.D*(self.K-1) - 1) / self.S + 1).astype(int)
#         self.Ly = np.floor((self.Y + 2*self.P - self.D*(self.K-1) - 1) / self.S + 1).astype(int)
        
#         # This masks is used to weight the patches when unpatchifying to allow for overlapping patches
#         self.mask = torch.ones_like(x).to(x.device)
                        
#         x = F.unfold(x, kernel_size=self.K, stride=self.S)
#         self.mask = F.unfold(self.mask, kernel_size=self.K, stride=self.S)
                
#         # x -> B (c*p*p) L
#         self.n_patches = x.shape[-1]
        
#         # Reshaping into the shape we want
#         if self.flatten_sequences:            
#             a = rearrange(x, '(n) (f x y) L -> (n L) f x y', x=self.K, y=self.K, n=self.n_scans)
#             self.mask = rearrange(self.mask, '(n) (f x y) L -> (n L) f x y', x=self.K, y=self.K, n=self.n_scans)
#         else:
#             a = rearrange(x, '(n) (f x y) L -> n L f x y', x=self.K, y=self.K, n=self.n_scans)
#             self.mask = rearrange(self.mask, '(n) (f x y) L -> n L f x y', x=self.K, y=self.K, n=self.n_scans)

#         if return_coordinates:
#             x = 1.0 * torch.arange(self.nx) - self.nx // 2
#             y = 1.0 * torch.arange(self.ny) - self.ny // 2
#             X, Y = torch.meshgrid(x, y, indexing='ij')
#             coordX = F.unfold(X[None, None, :, :], kernel_size=self.K, stride=self.S)            
#             coordY = F.unfold(Y[None, None, :, :], kernel_size=self.K, stride=self.S)

#             if self.flatten_sequences:            
#                 coordX = rearrange(coordX, '(n) (f x y) L -> (n L) f x y', x=self.K, y=self.K, n=self.n_scans)
#                 coordY = rearrange(coordY, '(n) (f x y) L -> (n L) f x y', x=self.K, y=self.K, n=self.n_scans)

#             patchX = torch.mean(coordX, dim=(-1, -2))
#             patchY = torch.mean(coordY, dim=(-1, -2))

#             XY = torch.cat([patchX, patchY], dim=-1)
            
#             return a, XY

#         else:
                
#             return a
    
#     def unpatchify(self, x, apodization=0, weight_type=None, weight_params=None):
#         """
#         Reconstructs the original image from patches.
#         Args:
#             x (torch.Tensor): The input tensor containing image patches with shape
#                               (n, L, f, x, y), where:
#                               - n: number of scans
#                               - L: number of patches
#                               - o: number of objects
#                               - f: number of frames (optional)
#                               - x, y: patch dimensions
#             apodization (int, optional): Number of pixels to apodize at the edges of the image. Default is 0.
#         Returns:
#             torch.Tensor: The reconstructed image tensor with shape 
#                           (n, o, f, x, y), where:
#                           - n: number of scans
#                           - o: number of objects
#                           - f: number of features (optional)
#                           - x, y: image dimensions
#         """

#         xndim = x.ndim
#         if xndim == 3:
#             if self.flatten_sequences:
#                 x2 = x.clone().unsqueeze(1)
#             else:
#                 x2 = x.clone().unsqueeze(2)
#             mask = self.mask[:, 0:1, :, :].clone()
#         else:
#             x2 = x.clone()
#             # mask = self.mask.clone()
#             mask = self.mask[:, 0:1, :, :].clone()

#         mask = mask.to(x.device)

#         if apodization > 0:            
#             x2 = x2[:, :, apodization:-apodization, apodization:-apodization]
#             mask = mask[:, :, apodization:-apodization, apodization:-apodization]
            
#         # Weight
#         if weight_type is None:
#             npix = x2.shape[-2]
#             self.weight = torch.ones((npix, npix)).to(x.device)
        
#         if weight_type == 'gaussian':
#             npix = x2.shape[-2]
#             xx = torch.linspace(-1, 1, npix, device=x.device)
#             yy = torch.linspace(-1, 1, npix, device=x.device)
#             X, Y = torch.meshgrid(xx, yy, indexing='ij')
#             R = torch.sqrt(X**2 + Y**2)
#             self.weight = torch.exp(-R**2 / weight_params)

#         if weight_type == 'cosine':
#             npix = x2.shape[-2]
#             npix_apod = min(weight_params, npix//2)

#             # Add a small value to avoid division by zero
#             win = np.hanning(2*npix_apod)
#             winOut = np.ones(npix)
#             winOut[0:npix_apod] = win[0:npix_apod]
#             winOut[-npix_apod:] = win[-npix_apod:]
#             weight = np.outer(winOut, winOut)
#             self.weight = torch.tensor(weight.astype('float32'), device=x.device)
        
#         # Reduce the size of the patch to account for the apodization
#         K = self.K - 2*apodization
        
#         # Compute the new size of the image
#         new_size_x = int((self.Lx - 1) * self.S + 1 - 2*self.P + self.D*(K-1))
#         new_size_y = int((self.Ly - 1) * self.S + 1 - 2*self.P + self.D*(K-1))        
                        
#         output_size = (new_size_x, new_size_y)
        
#         # Apply the weighting to the patches 
#         breakpoint()               
#         x2 *= self.weight[None, None, :, :]
#         mask *= self.weight[None, None, :, :]                
        
#         x2 = rearrange(x2, '(n L) f x y -> (n) (f x y) L', n=self.n_scans, L=self.n_patches)
#         mask = rearrange(mask, '(n L) f x y -> (n) (f x y) L', n=self.n_scans, L=self.n_patches)
                
#         x2 = F.fold(x2, output_size=output_size, kernel_size=K, stride=self.S)
#         mask = F.fold(mask, output_size=output_size, kernel_size=K, stride=self.S)

#         x2 = rearrange(x2, '(n) f x y -> n f x y', n=self.n_scans)
#         mask = rearrange(mask, '(n) f x y -> n f x y', n=self.n_scans)
                
#         final = x2 / (mask + 1e-10)

#         if xndim == 3:
#             final = final.squeeze(1)
                
#         return final
    
#     def unpatchify_old(self, x, apodization=0, weight_type=None, weight_params=None):
#         """
#         Reconstructs the original image from patches.
#         Args:
#             x (torch.Tensor): The input tensor containing image patches with shape
#                               (n, L, f, x, y), where:
#                               - n: number of scans
#                               - L: number of patches
#                               - o: number of objects
#                               - f: number of frames (optional)
#                               - x, y: patch dimensions
#             apodization (int, optional): Number of pixels to apodize at the edges of the image. Default is 0.
#         Returns:
#             torch.Tensor: The reconstructed image tensor with shape 
#                           (n, o, f, x, y), where:
#                           - n: number of scans
#                           - o: number of objects
#                           - f: number of features (optional)
#                           - x, y: image dimensions
#         """

#         xndim = x.ndim
#         if xndim == 3:
#             if self.flatten_sequences:
#                 x2 = x.clone().unsqueeze(1)
#             else:
#                 x2 = x.clone().unsqueeze(2)
#             mask = self.mask[:, 0:1, :, :].clone()
#         else:
#             x2 = x.clone()
#             mask = self.mask.clone()

#         mask = mask.to(x.device)

#         breakpoint()

#         if apodization > 0:
#             if self.flatten_sequences:
#                 x2 = x2[:, :, apodization:-apodization, apodization:-apodization]
#                 mask = mask[:, :, apodization:-apodization, apodization:-apodization]
#             else:
#                 x2 = x2[:, :, :, apodization:-apodization, apodization:-apodization]
#                 mask = mask[:, :, :, apodization:-apodization, apodization:-apodization]

#         # Weight
#         if weight_type is None:
#             npix = x2.shape[-2]
#             self.weight = torch.ones((npix, npix)).to(x.device)
        
#         if weight_type == 'gaussian':
#             npix = x2.shape[-2]
#             xx = torch.linspace(-1, 1, npix, device=x.device)
#             yy = torch.linspace(-1, 1, npix, device=x.device)
#             X, Y = torch.meshgrid(xx, yy, indexing='ij')
#             R = torch.sqrt(X**2 + Y**2)
#             self.weight = torch.exp(-R**2 / weight_params)

#         if weight_type == 'cosine':
#             npix = x2.shape[-2]
#             npix_apod = min(weight_params, npix//2)

#             # Add a small value to avoid division by zero
#             win = np.hanning(2*npix_apod)
#             winOut = np.ones(npix)
#             winOut[0:npix_apod] = win[0:npix_apod]
#             winOut[-npix_apod:] = win[-npix_apod:]
#             weight = np.outer(winOut, winOut)
#             self.weight = torch.tensor(weight.astype('float32'), device=x.device)
        
#         # Reduce the size of the patch to account for the apodization
#         K = self.K - 2*apodization
        
#         # Compute the new size of the image
#         new_size_x = int((self.Lx - 1) * self.S + 1 - 2*self.P + self.D*(K-1))
#         new_size_y = int((self.Ly - 1) * self.S + 1 - 2*self.P + self.D*(K-1))        
                        
#         output_size = (new_size_x, new_size_y)
        
#         # Apply the weighting to the patches                
#         if self.flatten_sequences:            
#             x2 *= self.weight[None, None, :, :]
#             mask *= self.weight[None, None, :, :]
#         else:            
#             x2 *= self.weight[None, None, None, :, :]
#             mask *= self.weight[None, None, None, :, :]
                
#         if self.flatten_sequences:
#             x2 = rearrange(x2, '(n L) f x y -> (n) (f x y) L', n=self.n_scans, L=self.n_patches)
#             mask = rearrange(mask, '(n L) f x y -> (n) (f x y) L', n=self.n_scans, L=self.n_patches)
#         else:
#             x2 = rearrange(x2, 'n L f x y -> (n) (f x y) L')
#             mask = rearrange(mask, 'n L f x y -> (n) (f x y) L')
        
#         x2 = F.fold(x2, output_size=output_size, kernel_size=K, stride=self.S)
#         mask = F.fold(mask, output_size=output_size, kernel_size=K, stride=self.S)

#         x2 = rearrange(x2, '(n) f x y -> n f x y', n=self.n_scans)
#         mask = rearrange(mask, '(n) f x y -> n f x y', n=self.n_scans)
                
#         final = x2 / (mask + 1e-10)

#         if xndim == 3:
#             final = final.squeeze(1)
                
#         return final
    
if __name__ == '__main__':
    n_scans = 1    
    n_frames = 12
    n_x = 80
    n_y = 80
    
    frames = torch.randn(n_scans, n_frames, n_x, n_y)

    print(' 4D-------------------')

    p = Patchify4D()

    print('Flatten sequence = True')
    print('-------------------------')

    patches = p.patchify(frames, patch_size=20, stride_size=10, flatten_sequences=True)
    print(f'  - Original : {frames.shape}')
    print(f'  - Patches : {patches.shape}')

    out = p.unpatchify(patches, apodization=3)    
    print(f'  - Reconstructed : {out.shape}')

    out = p.unpatchify(patches[:, 0:1, :, :], apodization=3)    
    print(f'  - Reconstructed (single): {out.shape}')

    out = p.unpatchify(patches[:, 0, :, :], apodization=3)    
    print(f'  - Reconstructed (single): {out.shape}')


    print('Flatten sequence = False')
    print('-------------------------')
    patches = p.patchify(frames, patch_size=20, stride_size=10, flatten_sequences=False)
    
    print(f'  - Original : {frames.shape}')
    print(f'  - Patches : {patches.shape}')
    
    out = p.unpatchify(patches, apodization=3)
    
    print(f'  - Reconstructed : {out.shape}')

    print(' 5D-------------------')

    n_scans = 1    
    n_frames = 12
    n_times = 2
    n_x = 80
    n_y = 80
    
    frames = torch.randn(n_scans, n_times, n_frames, n_x, n_y)

    p = Patchify4D()

    print('Flatten sequence = True')
    print('-------------------------')

    patches = p.patchify(frames, patch_size=20, stride_size=10, flatten_sequences=True)
    print(f'  - Original : {frames.shape}')
    print(f'  - Patches : {patches.shape}')

    out = p.unpatchify(patches, apodization=3)    
    print(f'  - Reconstructed : {out.shape}')

    out = p.unpatchify(patches[:, 0:1, :, :], apodization=3)    
    print(f'  - Reconstructed (single): {out.shape}')


    print('Flatten sequence = False')
    print('-------------------------')
    patches = p.patchify(frames, patch_size=20, stride_size=10, flatten_sequences=False)
    
    print(f'  - Original : {frames.shape}')
    print(f'  - Patches : {patches.shape}')
    
    out = p.unpatchify(patches, apodization=3)
    
    print(f'  - Reconstructed : {out.shape}')