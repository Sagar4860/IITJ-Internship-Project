import cv2
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont,ImageOps
import time
from PIL import Image, ImageDraw, ImageFont

import re
from wordfreq import top_n_list


from PIL import Image, ImageEnhance, ImageFilter
import random
import io
import os
import numpy as np


def generate_mask(clean_img, noisy_img):
    # Resize noisy image to match clean image dimensions (if needed)
    if clean_img.shape != noisy_img.shape:
        noisy_img = cv2.resize(noisy_img, (clean_img.shape[1], clean_img.shape[0]))
        # If clean_img is color and noisy_img is grayscale, convert noisy_img to color
        if len(clean_img.shape) == 3 and len(noisy_img.shape) == 2:
            noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2BGR)
        elif len(clean_img.shape) == 2 and len(noisy_img.shape) == 3:
            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2BGR)

    diff = cv2.absdiff(clean_img, noisy_img)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    # Convert single-channel binary mask to RGB
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask_rgb



def generate_corrupted_segmentation_mask(clean_mask, noisy_mask):
  
    if clean_mask is None or noisy_mask is None:
        print("❌ One or both masks are None.")
        return None

    if clean_mask.shape != noisy_mask.shape:
        print("❌ Masks have different shapes.")
        return None

    # Invert noisy mask so that noise becomes 0 (black)
    inverse_noisy_mask = cv2.bitwise_not(noisy_mask)

    # Apply bitwise AND to keep only clean text, remove noisy pixels
    corrupted_mask = cv2.bitwise_and(clean_mask, inverse_noisy_mask)

    return corrupted_mask

from PIL import Image
import numpy as np

def get_ism_from_clean_image(clean_image_path, threshold=200):
    # Load and convert to grayscale
    image = Image.open(clean_image_path).convert("L")

    # Convert to NumPy
    image_np = np.array(image)

    # Thresholding: text → white, background → black
    ism_np = np.where(image_np < threshold, 255, 0).astype(np.uint8)

    # Return as RGB NumPy array (3 channels)
    ism_rgb = cv2.cvtColor(ism_np, cv2.COLOR_GRAY2RGB)
    return ism_rgb


metadata = []

IGNCA_dir="/DATA1/ocrteam/word_restoration/reconstruction/IGNCA_Dataset"
df = pd.read_csv(os.path.join(IGNCA_dir, "meta_data.csv"))

def generate_dataset(
    phrases,
    language,
    clean_output_dir,
    noisy_output_dir,
    bitmask_output_dir,
    ISM_output_dir,
    CSM_output_dir,
):
    # print(phrases)
    cnt=2000
    for index,row in phrases.iterrows():
        gt_word=row["text"]
        if len(gt_word)>3 and index>6700:
            print(index)
            clean_ignca_path=os.path.join(IGNCA_dir,row["clean_path"] )
            noisy_ignca_path=os.path.join(IGNCA_dir,row["noisy_path"] )
            clean_img = cv2.imread(clean_ignca_path)
            clean_path=os.path.join(clean_output_dir,f"{cnt}.png")
            cv2.imwrite(clean_path,clean_img)
            
            noisy_img = cv2.imread(noisy_ignca_path)
            noisy_path=os.path.join(noisy_output_dir,f"{cnt}.png")
            cv2.imwrite(noisy_path,noisy_img)
            
            #masks
            bitmask_path = os.path.join(bitmask_output_dir, f"{cnt}.png")
            ISM_path=os.path.join(ISM_output_dir, f"{cnt}.png")
            CSM_path=os.path.join(CSM_output_dir, f"{cnt}.png")
            
            image = cv2.imread(clean_path)
            ISM_image = get_ism_from_clean_image(clean_path)
            cv2.imwrite(ISM_path,ISM_image)
            
            

            mask = generate_mask(image, noisy_img)
            cv2.imwrite(bitmask_path, mask)
            CSM_image = generate_corrupted_segmentation_mask(ISM_image, mask)
            cv2.imwrite(CSM_path, CSM_image)
            
            
            
            metadata.append({
                    "code":f"{cnt:05d}",
                    "word": gt_word,
                    "language": language,
                    "word_type":"Scene Text image",
                    "text_color": "IGNCA",
                    "background_color": "IGNCA",
                    "font_size": "IGNCA",
                    "font_family": "IGNCA",
                    "font_style": "IGNCA",
                    "font_path": "IGNCA",
                    "noise_type1": "IGNCA",
                    "noise_type2": "IGNCA",
                    "noise_type3": "IGNCA",
                    "angle": "IGNCA",
                    "clean_path": clean_path,
                    "noisy_path": noisy_path,
                    "binary_mask_path": bitmask_path,
                    "ISM_mask_path":ISM_path,
                    "CSM_mask_path":CSM_path
                })

            print(f"✅ {cnt} done")
            cnt += 1
            if cnt>2999:
                break

languages=["Sanskrit"]


for lang in languages: # Upload fonts here
    output_root = f'/DATA1/ocrteam/word_restoration/reconstruction/test2/{lang}_sample_test_dataset'
    print(f"generation Intialized for {output_root}")
    os.makedirs(output_root, exist_ok=True)
    clean_dir = os.path.join(output_root, 'clean')
    noisy_dir = os.path.join(output_root, 'noisy')
    bitmask_dir = os.path.join(output_root, 'bitmask')
    ISM_dir = os.path.join(output_root, 'Intact Segment Mask')
    CSM_dir=os.path.join(output_root, 'Corrupted Segment Mask')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(bitmask_dir, exist_ok=True)
    os.makedirs(ISM_dir, exist_ok=True)
    os.makedirs(CSM_dir, exist_ok=True)
    generate_dataset(
        df, 
        language=lang,
        clean_output_dir=clean_dir,
        noisy_output_dir=noisy_dir,
        bitmask_output_dir=bitmask_dir,
        ISM_output_dir=ISM_dir,
        CSM_output_dir=CSM_dir,
        )




output_root_csv = '/DATA1/ocrteam/word_restoration/reconstruction/test2'
metadata_csv_path = os.path.join(output_root_csv, "IGNCA_test_image.csv")
df = pd.DataFrame(metadata)
df.to_csv(metadata_csv_path, index=False)
print(f"📄 Metadata saved to {metadata_csv_path}")