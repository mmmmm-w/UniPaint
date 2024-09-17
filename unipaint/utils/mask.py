import numpy as np
import torch

class StaticRectangularMaskGenerator:
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
    
class MovingRectangularMaskGenerator:
    def __init__(self, rect_height_range, rect_width_range) -> None:
        """
        Initialize the MovingRectangularMaskGenerator with random rectangle sizes and movement.

        Args:
        - rect_height_range (tuple): A tuple (min_height, max_height) defining the range for the height of the rectangle.
        - rect_width_range (tuple): A tuple (min_width, max_width) defining the range for the width of the rectangle.
        """
        self.rect_height_range = rect_height_range
        self.rect_width_range = rect_width_range

    def __call__(self, control):
        """
        Generate a moving rectangular mask with random size and random movement.

        Args:
        - control (torch.Tensor): The control tensor to apply the mask to.
                                  Expected shape: (batch, channels, frames, height, width)

        Returns:
        - mask (torch.Tensor): The generated moving rectangular mask tensor.
                               Shape: (batch, channels, frames, height, width)
        """
        # Control tensor dimensions
        batch_size = control.shape[0]
        num_frames = control.shape[2]
        h = control.shape[-2]
        w = control.shape[-1]

        # Initialize a mask of ones (fully transparent to begin with)
        mask = torch.zeros_like(control)

        # Randomly choose rectangle sizes (height and width) for each sample in the batch
        rect_height = int((np.random.rand() * (self.rect_height_range[1] - self.rect_height_range[0]) + self.rect_height_range[0]) * h)
        rect_width = int((np.random.rand() * (self.rect_width_range[1] - self.rect_width_range[0]) + self.rect_width_range[0]) * w)

        # Ensure the starting point is valid
        x_start = np.random.randint(0, w - rect_width + 1)
        y_start = np.random.randint(0, h - rect_height + 1)

        # Ensure the ending point is valid
        x_end = np.random.randint(0, w - rect_width + 1)
        y_end = np.random.randint(0, w - rect_width + 1)

        # Interpolate the x and y positions for all frames
        x_positions = np.linspace(x_start, x_end, num_frames, dtype = int)
        y_positions = np.linspace(y_start, y_end, num_frames, dtype = int)

        # Apply the moving rectangle to each frame
        for frame_idx in range(num_frames):
            x_st = x_positions[frame_idx]
            y_st = y_positions[frame_idx]
            x_end_frame = min(x_st + rect_width, w)
            y_end_frame = min(y_st + rect_height, h)
            # Apply the mask by zeroing out the pixels inside the rectangle
            mask[:, :, frame_idx, y_st:y_end_frame, x_st:x_end_frame] = 1.0

        return mask
    
class MarginalMaskGenerator:
    def __init__(self, mask_l, mask_r, mask_t, mask_b) -> None:
        self.mask_l = mask_l  # Range for left mask boundary
        self.mask_r = mask_r  # Range for right mask boundary
        self.mask_t = mask_t  # Range for top mask boundary
        self.mask_b = mask_b  # Range for bottom mask boundary

    def __call__(self, control):
        mask = torch.zeros_like(control)
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
        mask[..., :t,:] = 1
        mask[..., -b:, :] = 1
        mask[..., :l] = 1
        mask[..., -r:] = 1

        return mask
    
class InterpolationMaskGenerator:
    def __init__(self, stride_range=(2, 5)) -> None:
        """
        Initialize the InterpolationMaskGenerator.
        This generator will mask out every n frame (i.e., 0 for unmasked frames, 1 for masked frames).
        """
        self.stride_range = stride_range

    def __call__(self, control):
        """
        Generate an interpolation mask that masks every second frame.

        Args:
        - control (torch.Tensor): The control tensor to apply the mask to.
                                  Expected shape: (batch, num_frames, channels, height, width)

        Returns:
        - mask (torch.Tensor): The generated interpolation mask tensor.
                               Shape: (batch, channels, frames, height, width)
        """
        # Initialize a mask of ones (fully masked to begin with)
        mask = torch.ones_like(control)
        batch_size = control.shape[0]

        for batch_idx in range(batch_size):
            # Randomly select a stride from the specified range
            stride = torch.randint(self.stride_range[0], self.stride_range[1] + 1, (1,)).item()

            # UnMask frames with the random stride, starting from frame index 0
            mask[batch_idx, :, 0::stride, :, :] = 0  # Unmask frames at intervals of the random stride

        return mask