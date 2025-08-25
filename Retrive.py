import tkinter as tk
from tkinter import filedialog, ttk
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image, ImageTk
import torch
from config import Config as C
from Data.Gallery import Gallery
from Model.Vit import VIT
import os
from pathlib import Path


gtPath = C.gtPath
queryPath = C.queryPath
transform = Compose([
        Resize((256, 128)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
modelPath = C.modelFile
galleryPath = C.galleryPath
gallery = Gallery()
gallery.load(galleryPath)
model       = VIT(imageHeight = 256, imageWidth = 128, nClasses = 751).to(device)
checkpoint  = torch.load(modelPath, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.to(device)

img_path = Path(queryPath) / "0015_c3s1_000826_00.jpg"
print(img_path)

queryImage = Image.open(img_path).convert("RGB")
queryTensor = transform(queryImage).unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    _, embedding = model(queryTensor)

results = gallery.search(embedding, topK = 10)

display = tk.Toplevel()
display.title("Retrieval Results")

canvas = tk.Canvas(display)
scrollbar = ttk.Scrollbar(display, orient="vertical", command=canvas.yview)
scroll_frame = ttk.Frame(canvas)
scroll_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

tk.Label(scroll_frame, text="Query Image", font=("Arial", 14, "bold")).pack(pady=5)
q_img_tk = ImageTk.PhotoImage(queryImage.resize((128, 256)))
tk.Label(scroll_frame, image=q_img_tk).pack(pady=5)

tk.Label(scroll_frame, text=f"Top-{10} Retrieved Images", font=("Arial", 14, "bold")).pack(pady=5)
retrieved_tk_images = []
for i, res in enumerate(results[0], start=1):
    file_path = os.path.join(gtPath, res["filename"])
    retrieved_img = Image.open(file_path).convert("RGB").resize((128, 256))
    r_img_tk = ImageTk.PhotoImage(retrieved_img)
    retrieved_tk_images.append(r_img_tk)  
    frame = ttk.Frame(scroll_frame)
    frame.pack(pady=5)
    tk.Label(frame, image=r_img_tk).pack(side="left")
    tk.Label(frame, text=f"Rank {i}\nLabel: {res['label']}\nScore: {res['score']:.4f}", font=("Arial", 10)).pack(side="left", padx=10)
display.mainloop()