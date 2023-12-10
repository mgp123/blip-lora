from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from transformers import BlipForConditionalGeneration, Trainer


class CaptionTrainer(Trainer):
    def compute_loss(self, model, batch):
        input_ids = batch.pop("input_ids")
        pixel_values = batch.pop("pixel_values")
        outputs = model(
            input_ids=input_ids, pixel_values=pixel_values, labels=input_ids
        )

        loss = outputs.loss
        return loss


class ImageCaptioningDataset(Dataset):
    def __init__(self, folder_path, processor):
        self.folder_path = Path(folder_path)
        self.dataset = pd.read_csv(self.folder_path / "metadata.csv").to_dict("records")
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(self.folder_path / item["file_name"]).convert("RGB")
        image = F.to_tensor(image)
        encoding = self.processor(
            images=image,
            text=item["caption"],
            padding="max_length",
            return_tensors="pt",
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding


def get_lora_model(model_name, r=16):
    model = BlipForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to("cuda")
    lora_config = LoraConfig(
        r=r,
        lora_alpha=r,
        target_modules=["*.query", "*.value"],
        lora_dropout=0.1,
        bias="none",
    )
    lora_model = get_peft_model(model, lora_config)
    return lora_model
