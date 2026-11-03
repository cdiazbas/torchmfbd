import torch

def warp(image, flow_AB, mode='bilinear'):
    """
    Warps an image using the provided flow field.

    Args:
        image (torch.Tensor): The input image tensor of shape (B, C, H, W), where B is the batch size,
                              C is the number of channels, H is the height, and W is the width.
        flow_AB (torch.Tensor): The flow field tensor of shape (B, 2, H, W) representing the displacement
                                vectors for each pixel in the image.
        device (torch.device): The device on which the tensors are allocated (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: The warped image tensor of shape (B, C, H, W).
    """
    
    B, C, H, W = image.size()

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)

    yy = torch.arange(0, H).view(-1,1).repeat(1,W)

    xx = xx.view(1,1,H,W).repeat(B,1,1,1)

    yy = yy.view(1,1,H,W).repeat(B,1,1,1)

    grid = torch.cat((xx,yy),1).float().to(image.device)

    vgrid = grid + flow_AB

    del xx, yy, grid

    # Normalise pixel coordinates with the convention of align_corners=False (pixel centres at
    # (2x+1)/W - 1), so that a zero flow is the identity. 2x/(W-1) - 1 belongs to align_corners=True;
    # mixed with False it magnifies the image by W/(W-1) and resamples it off the pixel centres.
    vgrid[:,0,:,:] = (2.0*vgrid[:,0,:,:] + 1.0) / W - 1.0

    vgrid[:,1,:,:] = (2.0*vgrid[:,1,:,:] + 1.0) / H - 1.0

    warped_image = torch.nn.functional.grid_sample(image, 
                                                   vgrid.permute(0,2,3,1), 
                                                   mode=mode, 
                                                   padding_mode='reflection', 
                                                   align_corners=False)

    return warped_image

def warp_affine(image, affine, mode='bilinear', padding_mode='reflection', return_grid=False):
    """
    Warps an image using the provided affine matrix.

    Args:
        image (torch.Tensor): The input image tensor of shape (H, W) where H is the height, and W is the width.
        flow_AB (torch.Tensor): The affine transformation matrix of shape (2, 3)        

    Returns:
        torch.Tensor: The warped image tensor of shape (H, W).
    """
        
    tmp = torch.nn.functional.affine_grid(affine, image.shape, align_corners=False)        
    warped_image = torch.nn.functional.grid_sample(image, 
                                                   tmp, 
                                                   mode=mode, 
                                                   padding_mode=padding_mode, 
                                                   align_corners=False)
    if return_grid:
        return warped_image, tmp
    else:
        return warped_image
