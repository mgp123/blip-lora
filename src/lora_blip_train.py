import argparse

import yaml
from torch.utils.data import random_split
from transformers import BlipProcessor, TrainingArguments

from utils import CaptionTrainer, ImageCaptioningDataset, get_lora_model


def train(
    dataset_path,
    model_name="Salesforce/blip-image-captioning-large",
    proccesor_name="Salesforce/blip-image-captioning-large",
):
    processor = BlipProcessor.from_pretrained(proccesor_name)
    dataset = ImageCaptioningDataset(dataset_path, processor)
    dataset_train, dataset_test = random_split(dataset, [0.9, 0.1])

    with open("training_args.yaml", "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    training_args = TrainingArguments(**train_config)

    lora_model = get_lora_model(model_name, r=3)
    trainer = CaptionTrainer(
        lora_model,
        training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,  # currently not used
        data_collator=None,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train lora model for image captioning."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the directory containing the dataset.",
    )
    parser.add_argument(
        "--model_name",
        default="Salesforce/blip-image-captioning-large",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--proccesor_name",
        default="Salesforce/blip-image-captioning-large",
        help="Name of the proccesor to use.",
    )
    args = parser.parse_args()
    train(args.dataset_path, args.model_name, args.proccesor_name)
