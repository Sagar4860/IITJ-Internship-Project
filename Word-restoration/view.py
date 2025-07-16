import pandas as pd
from PIL import Image
import numpy as np
import os

# Load CSV
df = pd.read_csv("/DATA1/ocrteam/word_restoration/reconstruction/test2/ocr_test4_data.csv")

LANG_COL = 'language'

image_cols = [
    'clean_path',
    'noisy_path',
    'output_path',
    'ISM_mask_path',
    'CSM_mask_path',
    'binary_mask_path'
]

RESIZE = (1024, 256)
PADDING = 30

def load_image(path):
    img = Image.open(path).convert("RGB")
    if RESIZE:
        img = img.resize(RESIZE)
    return img

def get_padding(height, width=PADDING, color=(255, 255, 255)):
    return np.full((height, width, 3), color, dtype=np.uint8)

# Select 6 unique languages only
languages = df[LANG_COL].dropna().unique()
selected_langs = languages[:6]  # pick top 6

rows = []

for lang in selected_langs:
    lang_df = df[df[LANG_COL] == lang]
    if len(lang_df) == 0:
        continue
    sample = lang_df.sample(n=1).iloc[0]
    
    images = [load_image(sample[col]) for col in image_cols]
    np_images = [np.asarray(img) for img in images]
    h = np_images[0].shape[0]
    
    # Add horizontal padding between images
    padded_row = np_images[0]
    for img in np_images[1:]:
        padded_row = np.hstack((padded_row, get_padding(h), img))

    rows.append(padded_row)

# Add vertical padding between language rows
padded_final = rows[0]
for row_img in rows[1:]:
    vpad = get_padding(PADDING, width=row_img.shape[1])
    padded_final = np.vstack((padded_final, vpad, row_img))

final_pil = Image.fromarray(padded_final)
final_pil.save(f"./language_comparison_grid.png")
