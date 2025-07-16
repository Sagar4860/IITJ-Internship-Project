import string
from difflib import SequenceMatcher
import easyocr
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
trocr_model.eval()

def get_trocr_text(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
        return trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        print(f"[TrOCR Error]: {e}")
        return ''

def get_tesseract_text(image_path, lang_code):
    try:
        return pytesseract.image_to_string(Image.open(image_path), lang=lang_code).strip()
    except:
        return '#####'

reader_cache = {}

def get_easyocr_text(image_path, lang_code):
    try:
        if lang_code not in reader_cache:
            reader_cache[lang_code] = easyocr.Reader([lang_code], gpu=True)
        result = reader_cache[lang_code].readtext(image_path, detail=0)
        return ' '.join(result).strip()
    except:
        return ''

def char_accuracy(gt, pred):
    return SequenceMatcher(None, gt, pred).ratio()

def clean_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

def normalize(text):
    return ''.join(clean_text(text))
