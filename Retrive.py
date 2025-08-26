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

img_path = Path(queryPath) / "1484_c6s3_084942_00.jpg"
print(img_path)

queryImage = Image.open(img_path).convert("RGB")
queryTensor = transform(queryImage).unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    _, embedding = model(queryTensor)

results = gallery.search(embedding, topK = 10)


root = tk.Tk()
root.withdraw()  # hide root window

display = tk.Toplevel(root)
display.title("Retrieval Results")
display.protocol("WM_DELETE_WINDOW", root.quit)
# Set window size (90% of screen width, 80% height)
screen_w = display.winfo_screenwidth()
screen_h = display.winfo_screenheight()
win_w, win_h = int(screen_w * 0.9), int(screen_h * 0.8)
x = (screen_w - win_w) // 2
y = (screen_h - win_h) // 4
display.geometry(f"{win_w}x{win_h}+{x}+{y}")
# Scrollable canvas
canvas = tk.Canvas(display)
scrollbar_y = ttk.Scrollbar(display, orient="vertical", command=canvas.yview)
scrollbar_x = ttk.Scrollbar(display, orient="horizontal", command=canvas.xview)
scroll_frame = ttk.Frame(canvas)
scroll_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all"),
        xscrollcommand=scrollbar_x.set,
        yscrollcommand=scrollbar_y.set
    )
)
canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
canvas.pack(side="left", fill="both", expand=True)
scrollbar_y.pack(side="right", fill="y")
scrollbar_x.pack(side="bottom", fill="x")
# Title
tk.Label(scroll_frame, text=f"Top-{len(results[0])} Retrieved Images",
         font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=5, pady=10)
retrieved_tk_images = []
for i, res in enumerate(results[0], start=1):
    file_path = os.path.join(gtPath, res["filename"])
    retrieved_img = Image.open(file_path).convert("RGB").resize((128, 256))
    r_img_tk = ImageTk.PhotoImage(retrieved_img)
    retrieved_tk_images.append(r_img_tk)  # keep ref
    # row and column calculation (2 rows × 5 cols)
    row = (i - 1) // 5 + 1   # +1 because row=0 has the title
    col = (i - 1) % 5
    frame = ttk.Frame(scroll_frame)
    frame.grid(row=row, column=col, padx=20, pady=20)
    tk.Label(frame, image=r_img_tk).pack()
    tk.Label(frame, text=f"Rank {i}\nLabel: {res['label']}\nScore: {res['score']:.4f}",
             font=("Arial", 10)).pack()
root.mainloop()