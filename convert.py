import torch
from safetensors.torch import save_file

# Load the checkpoint
checkpoint = torch.load('./checkpoints/sd-lora-epoch=71-val_loss=0.01.ckpt', map_location='cuda')

# Extract the state_dict
state_dict = checkpoint.get('state_dict', checkpoint)

# Save in safetensors format
save_file(state_dict, "./safetensors/gg.pth")
