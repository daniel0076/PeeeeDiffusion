import torch
import pytorch_lightning as pl
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from pathlib import Path
from main import StableDiffusionLoRA
import  os
#model_id = "CompVis/stable-diffusion-v1-4"



model_id = "./checkpoints/"
device = "cuda"
# Load the fine-tuned model
module = StableDiffusionLoRA.load_from_hyperparameters('./checkpoints/hparams.yaml')
# breakpoint()
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to(device)

# Now generate an image with the fine-tuned model
prompt = "cute, girl, drawing"
guidance_scales = [5, 7.5, 10]
for guidance_scale in guidance_scales:
    image = module.pipeline(prompt, num_inference_steps=25, guidance_scale=guidance_scale).images[0]

    # Save the generated image

    image.save(
        os.path.join( 'outputs',
            ''.join(prompt.split(",")) + '_'+ str(guidance_scale)+".png")
    )
