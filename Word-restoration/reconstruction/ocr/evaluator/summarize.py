import pandas as pd
import yaml

def summarize_ocr_results():
    with open("./config/ocr.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    easyocr_lang_map = cfg["easyocr_lang_map"]
    tesse_lang_map = cfg["tesseract_lang_map"]
    trocr_supported = set(cfg["trocr_supported_languages"])
    use_easyocr = cfg["ocr"]["use_easyocr"]
    use_tesseract = cfg["ocr"]["use_tesseract"]
    use_trocr = cfg["ocr"]["use_trocr"]
    output_csv = cfg["paths"]["output_csv"]

    df = pd.read_csv(output_csv)
    ocr_support = {
        "tesseract": set(tesse_lang_map.keys()) if use_tesseract else set(),
        "easyocr": set(easyocr_lang_map.keys()) if use_easyocr else set(),
        "trocr": trocr_supported if use_trocr else set()
    }

    print("\n🔍 Average Character-Level Accuracy by Language & OCR Tool:\n")
    for lang in df['language'].unique():
        print(f"Language: {lang}")
        lang_df = df[df['language'] == lang]

        for tool in ['tesseract', 'easyocr', 'trocr']:
            if lang not in ocr_support.get(tool, set()):
                print(f"  {tool.capitalize():<10} | ❌ Not Supported")
                continue

            acc_clean = lang_df[f"{tool}_acc_clean"].dropna().mean()
            acc_noisy = lang_df[f"{tool}_acc_noisy"].dropna().mean()
            acc_output = lang_df[f"{tool}_acc_output"].dropna().mean()
            print(f"  {tool.capitalize():<10} | Clean: {acc_clean:.2f}% | Noisy: {acc_noisy:.2f}% | Output: {acc_output:.2f}%")
        print("-" * 50)
