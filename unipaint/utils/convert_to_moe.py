import torch.nn as nn
import copy
import diffusers

def deterministic_gate(task_name):
    task_to_ffn_mapping = {
        "inpaint": 0,
        "outpaint": 1,
        "interpolation": 2,
        # Add more task-to-expert mappings here
    }
    return task_to_ffn_mapping.get(task_name, 0)  # Default to 0 if task_name not found

class MoEFFN(nn.Module):
    def __init__(self, num_tasks, original_ffn):
        super(MoEFFN, self).__init__()
        # Create a list of experts (one for each task), copying the structure
        self.gate = deterministic_gate
        self.experts = nn.ModuleList([
            nn.Sequential(
                copy.deepcopy(original_ffn.net[0]),  # Copy the original GEGLU (or other) activation layer
                copy.deepcopy(original_ffn.net[1]),  # Copy the original dropout
                copy.deepcopy(original_ffn.net[2])   # Copy the final Linear layer
            ) for _ in range(num_tasks)
        ])
    
    def forward(self, x, task_name):
        # Choose the appropriate expert based on the task_id
        task_id = self.gate(task_name)
        return self.experts[task_id](x)

def replace_ffn_with_moeffn(model, num_tasks, motion_module_name="motion_modules"):
    # Traverse all modules in the model
    for name, module in model.named_modules():
        if motion_module_name in name:
            # Look for 'ff' in the module name, which corresponds to the FeedForward block
            if isinstance(module, diffusers.models.attention.FeedForward):
                print(f"Replacing {name}.net with MoEFFN")
                # Replace the 'net' (which is a ModuleList) with the MoEFFN
                original_ffn = module  # Store the original FFN
                setattr(module, 'net', MoEFFN(num_tasks, original_ffn))  # Replace with MoEFFN
    return model