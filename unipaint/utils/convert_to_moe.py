import torch.nn as nn
import copy
import diffusers
from contextlib import contextmanager

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