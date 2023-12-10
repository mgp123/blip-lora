import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import functional as F
from transformers import BlipForConditionalGeneration, BlipProcessor


def caption(image, p, m) -> str:
    inputs = p(image, return_tensors="pt").to("cuda", torch.float16)
    out = m.generate(**inputs)
    return p.decode(out[0], skip_special_tokens=True)


def main(images_path, model_names):
    proc_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(proc_name)

    images_path = Path(images_path)

    images_list = list(images_path.glob("*.png"))

    images_list = random.sample(images_list, 3)
    images = [Image.open(image_path).convert("RGB") for image_path in images_list]

    # merge images into a big image
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    # plot images with their associated captions in a grid
    plt.figure(figsize=(12, 12))
    plt.imshow(new_im)
    average_height = total_height // len(images)
    colors = ["red", "green", "blue", "yellow", "orange", "purple"]
    text_offsets = [0] * len(images)
    for j, model_name in enumerate(model_names):
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(
            "cuda", torch.float16
        )
        for i, image in enumerate(images):
            caption_text = caption(image, processor, model)
            x_position = max_width + 10
            y_position = average_height * i + text_offsets[i] + average_height // 3

            text_obj = plt.text(
                x_position,
                y_position,
                caption_text,
                wrap=True,
                bbox=dict(facecolor=colors[j % len(colors)], alpha=0.5),
            )

            text_bbox = text_obj.get_window_extent(
                renderer=plt.gcf().canvas.get_renderer()
            )
            text_height = text_bbox.ymax - text_bbox.ymin
            text_offsets[i] += text_height // 2 + 10

        plt.text(
            -200,
            0 + j * 75,
            model_name,
            ha="center",
            va="center",
            bbox=dict(facecolor=colors[j % len(colors)], alpha=0.5),
        )

    plt.axis("off")

    plt.tight_layout()
    plt.savefig("caption_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot captions for images different models"
    )
    parser.add_argument(
        "--images_path",
        required=True,
        help="Path to the directory containing the images.",
    )
    # list of models to test
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=[
            "Salesforce/blip-image-captioning-large",
        ],
        help="Name of the model to use.",
    )

    args = parser.parse_args()
    main(args.images_path, args.model_names)
