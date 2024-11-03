import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from typing import List, Optional, Union, Dict, Any
import wandb

class StableDiffusionDataset(Dataset):
    def __init__(self, image_paths, prompts, tokenizer, image_size=512):
        self.image_paths = image_paths
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.image_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.image_transforms(image)

        prompt = self.prompts[idx]
        encoded = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "prompt": prompt
        }

class StableDiffusionLoRA(pl.LightningModule):
    def __init__(
        self,
        model_id: str = "CompVis/stable-diffusion-v1-4",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        target_modules: Optional[Union[List[str], str]] = None,
        layer_types: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
        sample_prompts: Optional[list] = None
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id)
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        # print stable diffusion model architecture
        # Freeze VAE and text encoder
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Setup UNet with custom LoRA
        self.unet = self.pipeline.unet
        print(self.unet)
        #write what it has been printed intofile:
        with open("unet.txt", "w") as file:
            file.write(str(self.unet))
        target_modules = self.get_target_modules(
            target_modules, layer_types, layer_indices
        )

        print(f"Applying LoRA to modules: {target_modules}")

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.unet = get_peft_model(self.unet, lora_config)

        # Sample prompts for validation
        self.sample_prompts = sample_prompts or ["A futuristic cityscape at night"]

        # Store configuration
        self.learning_rate = learning_rate

    def get_target_modules(
        self,
        target_modules: Optional[Union[List[str], str]],
        layer_types: Optional[List[str]],
        layer_indices: Optional[List[int]]
    ) -> List[str]:
        """Get list of target modules based on selection criteria."""
        if target_modules is not None:
            if isinstance(target_modules, str):
                return [target_modules]
            return target_modules

        attention_modules = []

        def get_attention_layers(model, prefix=""):
            for name, module in model.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                if any(t in module.__class__.__name__.lower() for t in ['attention', 'transformer']):
                    attention_modules.append({
                        'name': full_name,
                        'type': module.__class__.__name__,
                        'module': module
                    })

                if len(list(module.children())) > 0:
                    get_attention_layers(module, full_name)

        get_attention_layers(self.unet)

        # Filter based on criteria
        if layer_types:
            attention_modules = [
                mod for mod in attention_modules
                if any(t.lower() in mod['type'].lower() for t in layer_types)
            ]

        if layer_indices:
            attention_modules = [
                mod for i, mod in enumerate(attention_modules)
                if i in layer_indices
            ]

        if not layer_types and not layer_indices:
            return ["to_q", "to_k", "to_v", "to_out.0"]

        target_names = set()
        for mod in attention_modules:
            module_base = mod['name']
            target_names.update([
                f"{module_base}.to_q",
                f"{module_base}.to_k",
                f"{module_base}.to_v",
                f"{module_base}.to_out.0"
            ])

        return list(target_names)

    def forward(self, batch):
        latents = self.vae.encode(
            batch["pixel_values"].to(self.device)
        ).latent_dist.sample() * 0.18215

        text_embeddings = self.text_encoder(
            batch["input_ids"].to(self.device)
        )[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.pipeline.scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=self.device
        )
        noisy_latents = self.pipeline.scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            text_embeddings
        ).sample

        return noise_pred, noise

    def training_step(self, batch, batch_idx):
        noise_pred, noise = self(batch)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noise_pred, noise = self(batch)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        self.log("val_loss", loss, prog_bar=True)

        if batch_idx == 0 and self.current_epoch % 10 == 0:
            self.generate_samples()

        return loss

    def generate_samples(self):
        self.unet.eval()
        with torch.no_grad():
            for idx, prompt in enumerate(self.sample_prompts):
                image = self.pipeline(
                    prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5
                ).images[0]

                self.logger.experiment.log({
                    f"sample_{idx}": wandb.Image(image, caption=prompt)
                })
        self.unet.train()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.unet.parameters(), lr=self.learning_rate)

class StableDiffusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_paths: list,
        prompts: list,
        #tokenizer_id: str = "CompVis/stable-diffusion-v1-4",
        tokenizer_id: str = "openai/clip-vit-large-patch14",
        batch_size: int = 1,
        num_workers: int = 4,
        image_size: int = 512
    ):
        super().__init__()
        self.image_paths = image_paths
        self.prompts = prompts
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage: Optional[str] = None):
        dataset_size = len(self.image_paths)
        train_size = int(0.9 * dataset_size)

        self.train_dataset = StableDiffusionDataset(
            self.image_paths[:train_size],
            self.prompts[:train_size],
            self.tokenizer,
            self.image_size
        )

        self.val_dataset = StableDiffusionDataset(
            self.image_paths[train_size:],
            self.prompts[train_size:],
            self.tokenizer,
            self.image_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

def main(config):
    # Example configurations for different layer targeting

    #Initialize data module
    data_module = StableDiffusionDataModule(
        image_paths=config["image_paths"],
        prompts=config["prompts"],
        batch_size=config["batch_size"]
    )

    # Initialize model
    model = StableDiffusionLoRA(
        model_id=config["model_id"],
        learning_rate=config["learning_rate"],
        sample_prompts=config["sample_prompts"],
        **layer_config
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="sd-lora-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3
    )

    # Setup logger
    wandb_logger = WandbLogger(project="sd-lora-lightning")

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        precision=16,
        gradient_clip_val=1.0
    )

    # Start training
    trainer.fit(model, data_module)

if __name__ == "__main__":

    layer_configs = {
        "cross_attention_only": {
            "layer_types": ["attention"],
            "layer_indices": None,
            "target_modules": None
        },
        "down_blocks_only": {
            "target_modules": [
                f"down_blocks.{i}.attentions.0.to_q" for i in range(3)
            ],
            "layer_types": None,
            "layer_indices": None
        },
        "first_few_layers": {
            "layer_indices": [0, 1, 2],
            "layer_types": None,
            "target_modules": None
        },
        "end_layers": {
            "layer_indices": [5, 6, 7],
            "layer_types": None,
            "target_modules": None
        }
    }

    selected_config = "cross_attention_only"
    layer_config = layer_configs[selected_config]

    config = {
        "model_id": "CompVis/stable-diffusion-v1-4",
        "image_paths": ["./dataset/Peeee/10_peeee/1.png"],  # Update paths
        "prompts": ["peeee"],     # Update prompts
        "batch_size": 1,
        "max_epochs": 100,
        "learning_rate": 1e-4,
        "sample_prompts": [
            "A magical forest at sunset"
        ],
        **layer_config # Unpack layer configuration
    }

    main(config)
