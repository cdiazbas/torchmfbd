import numpy as np
import torch.nn.functional as F
import torch
from einops import rearrange

class Patchify4D(object):
    def __init__(self):
        super().__init__()

    def patchify(self, x, patch_size=64, stride_size=64, flatten_sequences=True):
        """
        Splits the input tensor into patches.
        Args:
            x (torch.Tensor): Input tensor of shape (n_scans, n_frames, nx, ny).
            patch_size (int, optional): Size of each patch. Default is 64.
            stride_size (int, optional): Stride size for patch extraction. Default is 64.
        Returns:
            torch.Tensor: Tensor containing the patches with shape (n_scans, L, n_frames, patch_size, patch_size), where L is the number of patches extracted.
        """
                            
        self.n_scans, self.n_frames, self.nx, self.ny = x.shape

        self.output_size = (self.nx, self.ny)
        self.flatten_sequences = flatten_sequences

        # padding, kernel size, stride, dilation and size of the input
        self.P = 0.0
        self.K = patch_size
        self.S = stride_size
        self.X = self.nx
        self.D = 1.0
        
        # Compute the size of blocks that will be extracted using unfold
        self.L = np.floor((self.X + 2*self.P - self.D*(self.K-1) - 1) / self.S + 1).astype(int)                

        # This masks is used to weight the patches when unpatchifying to allow for overlapping patches
        self.mask = torch.ones_like(x).to(x.device)
                        
        x = F.unfold(x, kernel_size=self.K, stride=self.S)
        self.mask = F.unfold(self.mask, kernel_size=self.K, stride=self.S)
        
        # x -> B (c*p*p) L
        self.n_patches = x.shape[-1]
        
        # Reshaping into the shape we want
        if self.flatten_sequences:            
            a = rearrange(x, '(n) (f x y) L -> (n L) f x y', x=self.K, y=self.K, n=self.n_scans)
            self.mask = rearrange(self.mask, '(n) (f x y) L -> (n L) f x y', x=self.K, y=self.K, n=self.n_scans)
        else:
            a = rearrange(x, '(n) (f x y) L -> n L f x y', x=self.K, y=self.K, n=self.n_scans)
            self.mask = rearrange(self.mask, '(n) (f x y) L -> n L f x y', x=self.K, y=self.K, n=self.n_scans)
                
        return a
    
    def unpatchify(self, x, apodization=0):
        """
        Reconstructs the original image from patches.
        Args:
            x (torch.Tensor): The input tensor containing image patches with shape
                              (n, L, f, x, y), where:
                              - n: number of scans
                              - L: number of patches
                              - o: number of objects
                              - f: number of frames (optional)
                              - x, y: patch dimensions
            apodization (int, optional): Number of pixels to apodize at the edges of the image. Default is 0.
        Returns:
            torch.Tensor: The reconstructed image tensor with shape 
                          (n, o, f, x, y), where:
                          - n: number of scans
                          - o: number of objects
                          - f: number of features (optional)
                          - x, y: image dimensions
        """

        xndim = x.ndim
        if xndim == 3:
            if self.flatten_sequences:
                x = x.unsqueeze(1)
            else:
                x = x.unsqueeze(2)
            mask = self.mask[:, 0:1, :, :]
        else:
            mask = self.mask

        mask = mask.to(x.device)

        if apodization > 0:
            x = x[:, :, apodization:-apodization, apodization:-apodization]
            mask = mask[:, :, apodization:-apodization, apodization:-apodization]

        # Reduce the size of the patch to account for the apodization
        K = self.K - 2*apodization

        # Compute the new size of the image
        new_size = int((self.L - 1) * self.S + 1 - 2*self.P + self.D*(K-1))
                        
        output_size = (new_size, new_size)
                
        if self.flatten_sequences:
            x = rearrange(x, '(n L) f x y -> (n) (f x y) L', n=self.n_scans, L=self.n_patches)
            mask = rearrange(mask, '(n L) f x y -> (n) (f x y) L', n=self.n_scans, L=self.n_patches)
        else:
            x = rearrange(x, 'n L f x y -> (n) (f x y) L')
            mask = rearrange(mask, 'n L f x y -> (n) (f x y) L')
        
        x = F.fold(x, output_size=output_size, kernel_size=K, stride=self.S)
        mask = F.fold(mask, output_size=output_size, kernel_size=K, stride=self.S)

        x = rearrange(x, '(n) f x y -> n f x y', n=self.n_scans)
        mask = rearrange(mask, '(n) f x y -> n f x y', n=self.n_scans)
        
        final = x / mask

        if xndim == 3:
            final = final.squeeze(1)
                
        return final
    
if __name__ == '__main__':
    n_scans = 1    
    n_frames = 12
    n_x = 80
    n_y = 80
    
    frames = torch.randn(n_scans, n_frames, n_x, n_y)

    p = Patchify4D()

    patches = p.patchify(frames, patch_size=20, stride_size=10)

    out = p.unpatchify(patches, apodization=3)