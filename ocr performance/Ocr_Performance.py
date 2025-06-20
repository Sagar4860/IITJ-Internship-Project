import pandas as pd
import easyocr
import pytesseract
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
from difflib import SequenceMatcher

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# print("🚀 Using GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "❌ No GPU detected")

# Language mapping
language_code = {"English": "en", "Hindi": "hi", "Japanese": "ja","Arabic":"ar","French":"fr","Chinese":"ch_sim","German":"de"}
tesse_language_code = {"English": "eng", "Hindi": "hin", "Japanese": "jpn","Arabic":"ara","French":"fra","Chinese":"chi_sim","German":"deu"}
    
# TrOCR supported languages (only tested on English by Microsoft officially)
trocr_supported_languages = {"English"}

# Load TrOCR model and processor
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
# trocr_model.to(device)
trocr_model.eval()


# EasyOCR reader cache
reader_cache = {}




def get_reader(lang_code):
    if lang_code not in reader_cache:
        reader_cache[lang_code] = easyocr.Reader([lang_code], gpu=False)
    return reader_cache[lang_code]


def char_accuracy(gt, pred):
    return SequenceMatcher(None, gt, pred).ratio()

def get_text_easyocr(image_path, lang_code):
    try:
        reader = get_reader(lang_code)
        result = reader.readtext(image_path, detail=0)
        return ' '.join(result).strip()
    except:
        return ''

def get_text_tesseract(image_path, lang_code):
    try:
        return pytesseract.image_to_string(Image.open(image_path), lang=lang_code).strip()
    except:
        return ''

def get_text_trocr(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
        return trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        print(f"[TrOCR Error]: {e}")
        return ''


# Load metadata
metadata = pd.read_csv('/DATA1/ocrteam/word_restoration/reconstruction/data/metadata.csv')
metadata = metadata[['code','word','language', 'clean_path', 'noisy_path']]


# Add columns for each OCR tool
columns = [
    'tesseract_clean', 'tesseract_noisy',
    'easyocr_clean', 'easyocr_noisy',
    'trocr_clean', 'trocr_noisy',
    'tesseract_acc_clean', 'tesseract_acc_noisy',
    'easyocr_acc_clean', 'easyocr_acc_noisy',
    'trocr_acc_clean', 'trocr_acc_noisy',   
]
for col in columns:
    metadata[col] = None

# Loop through metadata
for idx, row in metadata.iterrows():
    lang = row['language']
    gt_word = str(row['word']).strip()
    clean_path = row['clean_path']
    noisy_path = row['noisy_path']
    lang_code = language_code.get(lang, 'en')
    tesse_lang_code=tesse_language_code.get(lang,'eng')

    # EasyOCR
    ec = get_text_easyocr(clean_path, lang_code)
    en = get_text_easyocr(noisy_path, lang_code)
    metadata.at[idx, 'easyocr_clean'] = ec
    metadata.at[idx, 'easyocr_noisy'] = en
    metadata.at[idx, 'easyocr_acc_clean'] = round(char_accuracy(gt_word, ec) * 100, 2)
    metadata.at[idx, 'easyocr_acc_noisy'] = round(char_accuracy(gt_word, en) * 100, 2)

    # Tesseract
    tc = get_text_tesseract(clean_path, tesse_lang_code)
    tn = get_text_tesseract(noisy_path, tesse_lang_code)
    metadata.at[idx, 'tesseract_clean'] = tc
    metadata.at[idx, 'tesseract_noisy'] = tn
    metadata.at[idx, 'tesseract_acc_clean'] = round(char_accuracy(gt_word, tc) * 100, 2)
    metadata.at[idx, 'tesseract_acc_noisy'] = round(char_accuracy(gt_word, tn) * 100, 2)

    # TrOCR (English only)
    if lang in trocr_supported_languages:
        trc = get_text_trocr(clean_path)
        trn = get_text_trocr(noisy_path)
        metadata.at[idx, 'trocr_clean'] = trc
        metadata.at[idx, 'trocr_noisy'] = trn
        metadata.at[idx, 'trocr_acc_clean'] = round(char_accuracy(gt_word, trc) * 100, 2)
        metadata.at[idx, 'trocr_acc_noisy'] = round(char_accuracy(gt_word, trn) * 100, 2)
        
    

# Save the output
metadata.to_csv("ocr_evaluation_comparison.csv", index=False)
print("✅ Evaluation results saved to ocr_evaluation_comparison.csv")

# Grouped average accuracy
print("\n🔍 Average Character-Level Accuracy by Language & OCR Tool:\n")

for lang in metadata['language'].unique():
    print(f"Language: {lang}")
    lang_df = metadata[metadata['language'] == lang]

    for ocr_tool in ['tesseract', 'easyocr', 'trocr']:
        if ocr_tool == 'trocr' and lang not in trocr_supported_languages:
            continue
        acc_clean_col = f'{ocr_tool}_acc_clean'
        acc_noisy_col = f'{ocr_tool}_acc_noisy'
        avg_clean = lang_df[acc_clean_col].dropna().mean()
        avg_noisy = lang_df[acc_noisy_col].dropna().mean()
        print(f"  {ocr_tool.capitalize():<10} | Clean: {avg_clean:.2f}% | Noisy: {avg_noisy:.2f}%")
    print("-" * 50)
