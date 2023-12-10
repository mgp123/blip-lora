import csv
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from PIL import Image, ImageTk


class ImageCaptioningApp:
    def __init__(self, root, image_folder):
        self.root = root
        self.image_folder = image_folder
        self.image_list = [
            f
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
        ]
        self.current_image_index = 0

        self.caption_entry = tk.Entry(root)
        self.caption_entry.pack()

        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.next_button = tk.Button(root, text="Next Image", command=self.next_image)
        self.next_button.pack()

        self.root.bind("<Return>", lambda event=None: self.save_caption())

        self.load_image()

    def load_image(self):
        image_path = os.path.join(
            self.image_folder, self.image_list[self.current_image_index]
        )
        image = Image.open(image_path)
        image.thumbnail((1000, 1000))
        photo = ImageTk.PhotoImage(image)
        self.canvas.config(width=photo.width(), height=photo.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def next_image(self):
        self.current_image_index += 1
        if self.current_image_index < len(self.image_list):
            self.caption_entry.delete(0, tk.END)
            self.load_image()
        else:
            self.root.quit()

    def save_caption(self):
        caption = self.caption_entry.get()
        image_name = self.image_list[self.current_image_index]

        p = Path(os.path.join(self.image_folder, "metadata.csv"))
        p.touch(exist_ok=True)

        if caption.strip() != "":
            with open(p, "a", newline="") as csvfile:
                fieldnames = ["file_name", "caption"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if os.path.getsize(p) == 0:  # File is empty, write header
                    writer.writeheader()

                writer.writerow({"file_name": image_name, "caption": caption})

        self.next_image()


def main():
    root = tk.Tk()
    root.title("Image Captioning App")

    image_folder = filedialog.askdirectory(title="Select Image Folder")

    if image_folder:
        app = ImageCaptioningApp(root, image_folder)
        root.mainloop()


if __name__ == "__main__":
    main()
