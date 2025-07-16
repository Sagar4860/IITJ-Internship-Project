import pandas as pd
import easyocr
import pytesseract
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
from difflib import SequenceMatcher
from IndicPhotoOCR.ocr import OCR
import string



ocr_system = OCR(device="gpu", verbose=True)
def ocr_indicphotoocr(image_path, lang="english"):
    try:
        # Get OCR prediction
        return ocr_system.recognise(image_path, lang).strip()
    except Exception as e:
        print(f"[Error in IndicPhotoOCR]: {e}")
        return ''

def clean_text(text):
    # Lowercase and remove punctuation
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

def char_accuracy(gt, pred):
    return SequenceMatcher(None, gt, pred).ratio()


# Load metadata
metadata = pd.read_csv('/DATA1/ocrteam/word_restoration/reconstruction/test/merged_output.csv')

# Add columns for each OCR tool
columns = [
    'indicocr_clean', 'indicocr_noisy',
    'indicocr_acc_clean', 'indicocr_acc_noisy'
]
output_base_path = "/DATA1/ocrteam/word_restoration/reconstruction/test"
for col in columns:
    metadata[col] = None
for lang in metadata['language'].unique():
        for i in range(1000):
            metadata.loc[metadata['code'] == i, 'output_path'] = os.path.join(output_base_path, f'{lang}_test_sample_dataset/output/{i:03d}.png')
# Loop through metadata
count = 0 
for idx, row in metadata.iterrows():
    print(count)
    lang = row['language']
    gt_word = str(row['word']).strip()
    clean_path = row['clean_path']
    noisy_path = row['noisy_path']
    output_path = row['output_path']

    #IndicOcr
    # IndicPhotoOCR (English & Hindi only)
    if lang in {"English", "Hindi"}:
        if lang=='Hindi':
            code="hindi"
        else:
            code="english"
        ipc = ocr_indicphotoocr(clean_path, code)
        ipn = ocr_indicphotoocr(noisy_path, code)
        ipo = ocr_indicphotoocr(output_path,code)
        
         # Clean and flatten to character level
        ipn = ''.join(clean_text(ipn))
        ipc = ''.join(clean_text(ipc))
        ipo = ''.join(clean_text(ipo))
        gt_word=''.join(clean_text(gt_word))
        metadata.at[idx, 'indicocr_clean'] = ipc
        metadata.at[idx, 'indicocr_noisy'] = ipn
        metadata.at[idx, 'indicocr_output'] = ipo
        metadata.at[idx, 'indicocr_acc_clean'] = round(char_accuracy(gt_word, ipc) * 100, 2)
        metadata.at[idx, 'indicocr_acc_noisy'] = round(char_accuracy(gt_word, ipn) * 100, 2)
        metadata.at[idx, 'indicocr_acc_outpur'] = round(char_accuracy(gt_word, ipo) * 100, 2)
        count+=1


# Save the output
metadata.to_csv("indic_ocr_evaluation_comparison.csv", index=False)
print("✅ Evaluation results saved to ocr_evaluation_comparison.csv")

# Grouped average accuracy
print("\n🔍 Average Character-Level Accuracy by Language & OCR Tool:\n")

for lang in metadata['language'].unique():
    print(f"Language: {lang}")
    lang_df = metadata[metadata['language'] == lang]

    for ocr_tool in ['indicocr']:
        # if ocr_tool == 'trocr' and lang not in trocr_supported_languages:
        #     continue
        acc_clean_col = f'{ocr_tool}_acc_clean'
        acc_noisy_col = f'{ocr_tool}_acc_noisy'
        avg_clean = lang_df[acc_clean_col].dropna().mean()
        avg_noisy = lang_df[acc_noisy_col].dropna().mean()
        print(f"  {ocr_tool.capitalize():<10} | Clean: {avg_clean:.2f}% | Noisy: {avg_noisy:.2f}%")
    print("-" * 50)
