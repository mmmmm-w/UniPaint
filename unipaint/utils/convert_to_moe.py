import torch.nn as nn
import copy
import diffusers
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    def __init__(self, num_experts, mask_channels=1):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts

        # First convolutional layer with downsampling
        self.conv1 = nn.Conv3d(
            in_channels=mask_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,  # Downsampling by a factor of 2
            padding=1
        )
        self.bn1 = nn.BatchNorm3d(32)
        
        # Second convolutional layer with downsampling
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,  # Further downsampling
            padding=1
        )
        self.bn2 = nn.BatchNorm3d(64)
        
        # Third convolutional layer with downsampling
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,  # Further downsampling
            padding=1
        )
        self.bn3 = nn.BatchNorm3d(128)
        
        # Adaptive pooling to reduce dimensions to 1x1x1
        self.pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        
        # Fully connected layer to output weights for experts
        self.fc = nn.Linear(128, num_experts)
        
    def forward(self, x):
        # x shape: [batch, frame, channel, height, width]
        # Rearrange dimensions to [batch, channel, frame, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        
        # Pass through convolutional layers with downsampling
        x = F.relu(self.bn1(self.conv1(x)))  # Downsampled by a factor of 2
        x = F.relu(self.bn2(self.conv2(x)))  # Downsampled by a factor of 2
        x = F.relu(self.bn3(self.conv3(x)))  # Downsampled by a factor of 2
        
        # Adaptive pooling
        x = self.pool(x)  # Output shape: [batch, 128, 1, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch, 128]
        
        # Fully connected layer
        x = self.fc(x)  # Output shape: [batch, num_experts]
        weights = F.softmax(x, dim=1)  # Apply softmax
        return weights

class MaskGatedMoEFFN(nn.Module):
    def __init__(self, num_experts, original_ffn, mask_channels=1):
        super(MoEFFN, self).__init__()
        self.num_experts = num_experts
        self.gate = GatingNetwork(num_experts=num_experts, mask_channels=mask_channels)
        
        # Create multiple experts, copying the original FFN structure
        self.experts = nn.ModuleList([
            nn.Sequential(
                copy.deepcopy(original_ffn.net[0]),  # First layer (e.g., GEGLU)
                copy.deepcopy(original_ffn.net[1]),  # Dropout
                copy.deepcopy(original_ffn.net[2])   # Final Linear layer
            ) for _ in range(num_experts)
        ])

    def forward(self, x, mask):
        """
        x: Input tensor of shape [batch_size, input_dim]
        mask: Video mask tensor of shape [batch_size, frame, channel, height, width]
        """
        # Get gating weights from the gating network
        weights = self.gate(mask)  # Shape: [batch_size, num_experts]
        
        # Pass input x through all experts and stack outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # Shape: [batch_size, num_experts, output_dim]
        
        # Reshape weights to align with expert_outputs
        weights = weights.unsqueeze(-1)  # Shape: [batch_size, num_experts, 1]
        
        # Compute weighted sum of expert outputs
        output = torch.sum(weights * expert_outputs, dim=1)  # Shape: [batch_size, output_dim]
        return output

@contextmanager
def task_context(task_name):
    global CURRENT_TASK_NAME
    CURRENT_TASK_NAME = task_name
    yield
    CURRENT_TASK_NAME = None

def get_task_name():
    return CURRENT_TASK_NAME

def deterministic_gate(task_name):
    task_to_ffn_mapping = {
        "inpaint": 0,
        "outpaint": 1,
        "interpolation": 2,
        # Add more task-to-expert mappings here
    }
    return task_to_ffn_mapping.get(task_name, 0)
    
class MoEFFN(nn.Module):
    def __init__(self, num_tasks, original_ffn):
        super(MoEFFN, self).__init__()
        # Copy dimensions and components from the original FFN
        self.gate = deterministic_gate
        
        # Create multiple experts (one for each task), copying the original FFN structure
        self.experts = nn.ModuleList([
            nn.Sequential(
                copy.deepcopy(original_ffn.net[0]),  # First GEGLU or other layer
                copy.deepcopy(original_ffn.net[1]),  # Dropout
                copy.deepcopy(original_ffn.net[2])   # Final Linear layer
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        task_name = get_task_name()
        task_id = self.gate(task_name)
        x = self.experts[task_id](x)
        return x

def replace_ffn_with_moeffn(model, num_tasks=3, motion_module_name="motion_modules"):
    # Collect the modules that need to be replaced
    modules_to_replace = []

    # Traverse all modules and collect the ones to be replaced
    for name, module in model.named_modules():
        if motion_module_name in name:
            if isinstance(module, diffusers.models.attention.FeedForward):
                print(f"Marked {name} for replacement with MoEFFN")
                modules_to_replace.append((name, module))

    # Now replace the collected modules after iteration
    for name, original_ffn in modules_to_replace:
        # Navigate to the parent module
        parent_module = model
        *parent_path, last_name = name.split('.')
        
        for attr in parent_path:
            parent_module = getattr(parent_module, attr)
        
        # Replace the FeedForward block with MoEFFN
        setattr(parent_module, last_name, MoEFFN(num_tasks, original_ffn))
    
    return model