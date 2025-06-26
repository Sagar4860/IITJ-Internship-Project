import pandas as pd
import os
from utils.all_ocr import *
import yaml

def run_ocr():
    with open("config/ocr.yaml", "r") as f:
        cfg = yaml.safe_load(f)

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

    metadata = pd.read_csv(metadata_path)
    metadata = metadata[['code','word',"noise_type1","noise_type2","noise_type3",'language', 'clean_path', 'noisy_path']]
    for col in [
        'tesseract_clean', 'tesseract_noisy','tesseract_output',
        'easyocr_clean', 'easyocr_noisy','easyocr_output',
        'trocr_clean', 'trocr_noisy','trocr_output',
        'tesseract_acc_clean', 'tesseract_acc_noisy','tesseract_acc_output',
        'easyocr_acc_clean', 'easyocr_acc_noisy','easyocr_acc_output',
        'trocr_acc_clean', 'trocr_acc_noisy',  'trocr_acc_output'
    ]:
        metadata[col] = None

    for i in range(num_samples):
        metadata.loc[metadata['code'] == i, 'output_path'] = os.path.join(output_base_path, f'{i:03d}.png')

    for idx, row in metadata.iterrows():
        lang = row['language']
        gt_word = str(row['word']).strip()
        clean_path = row['clean_path']
        noisy_path = row['noisy_path']
        output_path = row['output_path']

        if use_easyocr and lang in easyocr_lang_map:
            code = easyocr_lang_map[lang]
            ec = get_easyocr_text(clean_path, code)
            en = get_easyocr_text(noisy_path, code)
            eo = get_easyocr_text(output_path, code)
            metadata.at[idx, 'easyocr_clean'] = ec
            metadata.at[idx, 'easyocr_noisy'] = en
            metadata.at[idx, 'easyocr_output'] = eo
            metadata.at[idx, 'easyocr_acc_clean'] = round(char_accuracy(gt_word, ec) * 100, 2)
            metadata.at[idx, 'easyocr_acc_noisy'] = round(char_accuracy(gt_word, en) * 100, 2)
            metadata.at[idx, 'easyocr_acc_output'] = round(char_accuracy(gt_word, eo) * 100, 2)

        if use_tesseract and lang in tesse_lang_map:
            code = tesse_lang_map[lang]
            tc = get_tesseract_text(clean_path, code)
            tn = get_tesseract_text(noisy_path, code)
            to = get_tesseract_text(output_path, code)
            metadata.at[idx, 'tesseract_clean'] = tc
            metadata.at[idx, 'tesseract_noisy'] = tn
            metadata.at[idx, 'tesseract_output'] = to
            metadata.at[idx, 'tesseract_acc_clean'] = round(char_accuracy(gt_word, tc) * 100, 2)
            metadata.at[idx, 'tesseract_acc_noisy'] = round(char_accuracy(gt_word, tn) * 100, 2)
            metadata.at[idx, 'tesseract_acc_output'] = round(char_accuracy(gt_word, to) * 100, 2)

        if use_trocr and lang in trocr_supported:
            trc = get_trocr_text(clean_path)
            trn = get_trocr_text(noisy_path)
            tro = get_trocr_text(output_path)
            metadata.at[idx, 'trocr_clean'] = trc
            metadata.at[idx, 'trocr_noisy'] = trn
            metadata.at[idx, 'trocr_output'] = tro
            metadata.at[idx, 'trocr_acc_clean'] = round(char_accuracy(gt_word, trc) * 100, 2)
            metadata.at[idx, 'trocr_acc_noisy'] = round(char_accuracy(gt_word, trn) * 100, 2)
            metadata.at[idx, 'trocr_acc_output'] = round(char_accuracy(gt_word, tro) * 100, 2)

    metadata.to_csv(output_csv, index=False)
    print(f"✅ OCR results saved to {output_csv}")
