import pandas as pd
import easyocr
import pytesseract
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
from difflib import SequenceMatcher
import yaml

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Config values
easyocr_lang_map = cfg["easyocr_lang_map"]
tesse_lang_map = cfg["tesseract_lang_map"]
trocr_supported = set(cfg["trocr_supported_languages"])
use_easyocr = cfg["ocr"]["use_easyocr"]
use_tesseract = cfg["ocr"]["use_tesseract"]
use_trocr = cfg["ocr"]["use_trocr"]
metadata_path = cfg["paths"]["metadata_csv"]
output_base_path = cfg["paths"]["output_base_path"]
output_csv = cfg["paths"]["output_csv"]
num_samples = cfg["num_samples"]

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

metadata  = pd.read_csv(metadata_path)
# Load metadata
metadata = metadata[['code','word',"noise_type1","noise_type2","noise_type3",'language', 'clean_path', 'noisy_path']]

# Add columns for each OCR tool
columns = [
    'tesseract_clean', 'tesseract_noisy','tesseract_output',
    'easyocr_clean', 'easyocr_noisy','easyocr_output',
    'trocr_clean', 'trocr_noisy','trocr_output',
    'tesseract_acc_clean', 'tesseract_acc_noisy','tesseract_acc_output',
    'easyocr_acc_clean', 'easyocr_acc_noisy','easyocr_acc_output',
    'trocr_acc_clean', 'trocr_acc_noisy',  'trocr_acc_output' 
]
for col in columns:
    metadata[col] = None

for i in range(num_samples):
    metadata.loc[metadata['code'] == i, 'output_path'] = os.path.join(output_base_path, f'{i:03d}.png')


# Loop through metadata
for idx, row in metadata.iterrows():
    lang = row['language']
    gt_word = str(row['word']).strip()
    clean_path = row['clean_path']
    noisy_path = row['noisy_path']
    output_path = row['output_path']
    
    if use_easyocr and lang in easyocr_lang_map:
        code = easyocr_lang_map[lang]
        ec = get_text_easyocr(clean_path, code)
        en = get_text_easyocr(noisy_path, code)
        eo = get_text_easyocr(output_path, code)
        metadata.at[idx, 'easyocr_clean'] = ec
        metadata.at[idx, 'easyocr_noisy'] = en
        metadata.at[idx, 'easyocr_output'] = eo
        metadata.at[idx, 'easyocr_acc_clean'] = round(char_accuracy(gt_word, ec) * 100, 2)
        metadata.at[idx, 'easyocr_acc_noisy'] = round(char_accuracy(gt_word, en) * 100, 2)
        metadata.at[idx, 'easyocr_acc_output'] = round(char_accuracy(gt_word, eo) * 100, 2)

    # Tesseract
    if use_tesseract and lang in tesse_lang_map:
        code = tesse_lang_map[lang]
        tc = get_text_tesseract(clean_path, code)
        tn = get_text_tesseract(noisy_path, code)
        to = get_text_tesseract(output_path, code)
        metadata.at[idx, 'tesseract_clean'] = tc
        metadata.at[idx, 'tesseract_noisy'] = tn
        metadata.at[idx, 'tesseract_output'] = to
        metadata.at[idx, 'tesseract_acc_clean'] = round(char_accuracy(gt_word, tc) * 100, 2)
        metadata.at[idx, 'tesseract_acc_noisy'] = round(char_accuracy(gt_word, tn) * 100, 2)
        metadata.at[idx, 'tesseract_acc_output'] = round(char_accuracy(gt_word, to) * 100, 2)

    # TrOCR
    if use_trocr and lang in trocr_supported:
        trc = get_text_trocr(clean_path)
        trn = get_text_trocr(noisy_path)
        tro = get_text_trocr(output_path)
        metadata.at[idx, 'trocr_clean'] = trc
        metadata.at[idx, 'trocr_noisy'] = trn
        metadata.at[idx, 'trocr_output'] = tro
        metadata.at[idx, 'trocr_acc_clean'] = round(char_accuracy(gt_word, trc) * 100, 2)
        metadata.at[idx, 'trocr_acc_noisy'] = round(char_accuracy(gt_word, trn) * 100, 2)
        metadata.at[idx, 'trocr_acc_output'] = round(char_accuracy(gt_word, tro) * 100, 2)
        
# Save the output
metadata.to_csv("ocr_test_data_tesseract.csv", index=False)
print("✅ Evaluation results saved to ocr_evaluation_comparison.csv")

# Grouped average accuracy
print("\n🔍 Average Character-Level Accuracy by Language & OCR Tool:\n")


# OCR support maps
ocr_support = {
    "tesseract": set(tesse_lang_map.keys()) if use_tesseract else set(),
    "easyocr": set(easyocr_lang_map.keys()) if use_easyocr else set(),
    "trocr": trocr_supported if use_trocr else set()
}

for lang in metadata['language'].unique():
    print(f"Language: {lang}")
    lang_df = metadata[metadata['language'] == lang]

    for tool in ['tesseract', 'easyocr', 'trocr']:
        if lang not in ocr_support.get(tool, set()):
            print(f"  {tool.capitalize():<10} | ❌ Not Supported")
            continue

        acc_clean = lang_df[f"{tool}_acc_clean"].dropna().mean()
        acc_noisy = lang_df[f"{tool}_acc_noisy"].dropna().mean()
        acc_output = lang_df[f"{tool}_acc_output"].dropna().mean()

        print(f"  {tool.capitalize():<10} | Clean: {acc_clean:.2f}% | Noisy: {acc_noisy:.2f}% | Output: {acc_output:.2f}%")

    print("-" * 50)
