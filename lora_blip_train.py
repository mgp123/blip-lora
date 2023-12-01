from pathlib import Path
import pandas as pd
from peft import LoraConfig, get_peft_model
import torch
from transformers import  BlipProcessor, BlipForConditionalGeneration, TrainingArguments
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from transformers import Trainer
import yaml 
import argparse


class CaptionTrainer(Trainer):
    def compute_loss(self, model, batch):
        input_ids = batch.pop("input_ids")
        pixel_values = batch.pop("pixel_values")
        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids)
        
        loss = outputs.loss
        return loss

class ImageCaptioningDataset(Dataset):
    def __init__(self, folder_path, processor):
        self.folder_path = Path(folder_path)
        self.dataset = pd.read_csv(self.folder_path / "metadata.csv").to_dict('records')
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(self.folder_path / item["file_name"]).convert("RGB")
        image = F.to_tensor(image)
        encoding = self.processor(images=image, text=item["caption"], padding="max_length", return_tensors="pt")
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding


def train(dataset_path, model_name="Salesforce/blip-image-captioning-large", proccesor_name="Salesforce/blip-image-captioning-large"):
    processor = BlipProcessor.from_pretrained(proccesor_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
    dataset = ImageCaptioningDataset(dataset_path, processor)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["*.key", "*.query", "*.value"],
        lora_dropout=0.1,
        bias="none",
    )

    with open("training_args.yaml", "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    training_args = TrainingArguments(
        **train_config
    )

    lora_model = get_peft_model(model, lora_config)

    trainer = CaptionTrainer(
        lora_model,
        training_args,
        train_dataset=dataset,
        data_collator=None,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lora model for image captioning.")
    parser.add_argument("--dataset_path", required=True, help="Path to the directory containing the dataset.")
    parser.add_argument("--model_name", default="Salesforce/blip-image-captioning-large", help="Name of the model to use.")
    parser.add_argument("--proccesor_name", default="Salesforce/blip-image-captioning-large", help="Name of the proccesor to use.")
    args = parser.parse_args()
    train(args.dataset_path, args.model_name, args.proccesor_name)

