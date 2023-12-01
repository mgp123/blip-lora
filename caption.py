import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from torchvision.transforms import functional as F
import os
import argparse
from tqdm import tqdm
import pandas as pd

def generate_captions(frames_directory, output_file="metadata.csv"):

    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

    total_frames = len([f for f in os.listdir(frames_directory) if f.endswith(".png")])
    rows = []
    for filename in tqdm(sorted(os.listdir(frames_directory)), total=total_frames, desc="Generating Captions"):
        if filename.endswith(".png"):
            frame_path = os.path.join(frames_directory, filename)

            image = Image.open(frame_path).convert("RGB")
            image = F.to_tensor(image)
            inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)

            caption_ids = model.generate(**inputs)
            caption = processor.decode(caption_ids[0], skip_special_tokens=True)
            rows.append({"file_name": filename, "caption": caption})
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(frames_directory, output_file), index=False)
    print(f"Captions saved to {output_file} successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for frames in a directory.")
    parser.add_argument("--frames_directory", required=True, help="Path to the directory containing the frames.")
    parser.add_argument("--output_file", default="metadata.csv", help="Path to the file where captions will be saved.")

    args = parser.parse_args()
    generate_captions(args.frames_directory, args.output_file)
