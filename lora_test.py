import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

from PIL import Image


def caption(image, p, m) -> str:
    inputs = p(image, return_tensors="pt").to("cuda", torch.float16)
    out = m.generate(**inputs)
    return p.decode(out[0], skip_special_tokens=True)

proc_name = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(proc_name)

images_paths = [
    "output_frames/frame_0027.png", 
    "output_frames/frame_0925.png",
    "output_frames/frame_0982.png",
    "output_frames/frame_0154.png"
    ]
images = [Image.open(image_path).convert("RGB") for image_path in images_paths]

model_names = ["./futurama-lora-blip/checkpoint-22", "./futurama-lora-blip/checkpoint-40", "Salesforce/blip-image-captioning-large"]

# plot images with their associated captions in a grid
fig, axs = plt.subplots(len(images), 1)
plt.subplots_adjust(hspace=0.0, wspace=0.0)
colors = ["red", "green", "blue"]
fontsize = 6
for j, model_name in enumerate(model_names):
    model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
    for i, image in enumerate(images):
        axs[i].imshow(image)
        s = caption(image, processor, model)
        axs[i].text(image.width + 30, 50 + j*20*(fontsize), s ,fontsize=fontsize,  bbox = dict(facecolor = colors[j%3], alpha = 0.5))
        axs[i].axis('off')

    plt.text(- 540, 0 + j*20*(fontsize), model_name, ha='center', va='center', fontsize=fontsize, bbox = dict(facecolor = colors[j%3], alpha = 0.5))

# axs[i].text(image.width + 30, 40 + j*20*(fontsize), caption(image, processor, model) ,fontsize=fontsize,  bbox = dict(facecolor = colors[j%3], alpha = 0.5))

# add text to the left of plot

# change margins
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.tight_layout()
plt.savefig("caption_comparison.png")
#plt.show()


