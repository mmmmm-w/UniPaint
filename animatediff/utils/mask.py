import numpy as np
import torch

class RectangularMaskGenerator:
    def __init__(self, mask_l, mask_r, mask_t, mask_b) -> None:
        self.mask_l = mask_l  # Range for left mask boundary
        self.mask_r = mask_r  # Range for right mask boundary
        self.mask_t = mask_t  # Range for top mask boundary
        self.mask_b = mask_b  # Range for bottom mask boundary

    def __call__(self, control):
        mask = torch.ones_like(control)
        h = mask.shape[-2]
        w = mask.shape[-1]

        # Generate random boundaries
        l = np.random.rand() * (self.mask_l[1] - self.mask_l[0]) + self.mask_l[0]
        r = np.random.rand() * (self.mask_r[1] - self.mask_r[0]) + self.mask_r[0]
        t = np.random.rand() * (self.mask_t[1] - self.mask_t[0]) + self.mask_t[0]
        b = np.random.rand() * (self.mask_b[1] - self.mask_b[0]) + self.mask_b[0]

        # Convert to integer indices
        l, r, t, b = int(l * w), int(r * w), int(t * h), int(b * h)

        # Apply the mask
        mask[..., :t,:] = 0
        mask[..., -b:, :] = 0
        mask[..., :l] = 0
        mask[..., -r:] = 0

        return mask