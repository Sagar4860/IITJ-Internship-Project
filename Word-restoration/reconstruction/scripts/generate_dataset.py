import os
import random
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
from scripts.noise_generator import apply_noise
from utils import generate_corrupted_mask, generate_corrupted_segmentation_mask

class TextImageGenerator:
    def __init__(self, language='English', config=None):
        self.config = config
        self.fonts_dir = Path(os.path.join(self.config['fonts_dir'], language))
        self.output_data_dir = self.config['output_data_dir']
        self.clean_image_dir = os.path.join(self.output_data_dir, language, 'clean')
        self.noisy_image_dir = os.path.join(self.output_data_dir, language, 'noisy')
        self.clean_image_binary_mask_dir = os.path.join(self.output_data_dir, language, 'clean_binary')
        self.noisy_image_binary_mask_dir = os.path.join(self.output_data_dir, language, 'noisy_binary')
        self.noist_text_binary_mask_dir = os.path.join(self.output_data_dir, language, 'noisy_text')
        self.font_size_range = set(self.config['font_size_range'])
        self.rotate_prob = self.config['rotate_probability']
        self.rotate_range = set(self.config['rotate_range'])
        self.language = language
        self.metadata_path = os.path.join(self.output_data_dir, 'meta_data.csv')
        if os.path.exists(self.metadata_path):
            self.df = pd.read_csv(self.metadata_path)
        else:
            self.df =pd.DataFrame(self.config['meta_data'])

    def text_and_background_color(self, colors):
        light_colors = colors[0]['light_colors']
        dark_colors = colors[1]['dark_colors']
        if random.random() < 0.5:
            text_color = random.choice(dark_colors)
            background_color = random.choice(light_colors)
        else:
            text_color = random.choice(light_colors)
            background_color = random.choice(dark_colors)

        return text_color, background_color
    
    def process_noise_types(self, noise):
        text_noises = noise[0]['text_noise']
        image_artifcats = noise[1]['image_artifacts']
        environment_noises = noise[2]['environmental_noise']

        return text_noises, image_artifcats, environment_noises


    def generate_text_image(self, text, font_path, font_size, text_color, background_color, angle, padding=4):
        font = ImageFont.truetype(font_path, font_size)
        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)

        text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        offset = (padding - text_bbox[0], padding-text_bbox[1])

        clean_image = Image.new("RGB", (text_width, text_height), color=background_color)
        draw = ImageDraw.Draw(clean_image)
        draw.text(offset, text, font=font, fill=text_color)

        clean_binary_image = Image.new("RGB", (text_width, text_height), color="black")
        draw_ism = ImageDraw.Draw(clean_binary_image)
        draw_ism.text(offset, text, font=font, fill="white")

        if angle != 0:
            clean_image = clean_image.rotate(angle, expand=True, fillcolor=background_color)
            clean_binary_image = clean_binary_image.rotate(angle, expand=True, fillcolor=background_color)

        return clean_image, clean_binary_image
    

    def generate_images(self, corpus, number_samples):
        font_families = [f for f in self.fonts_dir.iterdir() if f.is_dir()]
        if not font_families:
            raise ValueError("No font family folders found.")

        padding = self.config['padding']
        text_noises, image_artifacts, environmental_noises = self.process_noise_types(self.config['noise'])
        
        for sample in range(number_samples):
            word = random.choice(corpus)
            angle = int(random.uniform(*self.rotate_range)) if random.random() <= self.rotate_prob else 0
            random_family = random.choice(font_families)
            fonts = list(random_family.glob("*.ttf")) + list(random_family.glob("*.otf"))

            if not fonts:
                print(f"No fonts found in {random_family}, skipping...")
                continue

            random_font = random.choice(fonts)
            text_color, background_color = self.text_and_background_color(self.config['colors'])
            font_name = random_font.stem.replace(" ", "_").split('-')[-1]
            font_size = random.randint(*self.font_size_range)

            filename = f"{sample:010}.jpg"
            
            clean_image, clean_binary_image = self.generate_text_image(
                text=word,
                font_path=random_font,
                font_size=font_size,
                text_color=text_color,
                background_color=background_color,
                angle=angle,
                padding=padding
            )

            text_noise = random.choice(text_noises)
            noisy_image = apply_noise(clean_image, text_noise)

            rand_val = random.random()
            if rand_val < 0.3:
                image_artifact = random.choice(image_artifacts)
                noisy_image = apply_noise(noisy_image, image_artifact)

            elif rand_val < 0.6:
                environmental_noise = random.choice(environmental_noises)
                noisy_image = apply_noise(noisy_image, environmental_noise)

            elif rand_val < 0.8:
                image_artifact = random.choice(image_artifacts)
                noisy_image = apply_noise(noisy_image, image_artifact)
                environmental_noises = [n for n in environmental_noises if n not in {'aliasing', 'jpeg compression'}]
                environmental_noise = random.choice(environmental_noises)
                noisy_image = apply_noise(noisy_image, environmental_noise)
            
            noisy_binary_image = generate_corrupted_mask(noisy_image, clean_image)
            noisy_text_binary_image = generate_corrupted_segmentation_mask(noisy_binary_image, clean_binary_image)

            self.df.loc[len(self.df)] = [
                filename,
                word,
                self.language,
                text_color,
                background_color,
                font_size,
                random_family.name,
                font_name,
                text_noise,
                image_artifact,
                environmental_noise,
                angle,
                clean_image_path,
                noisy_image_path,
                clean_binary_image_path,
                noisy_binary_image_path,
                noisy_text_binary_image_path
            ]
            
            clean_image_path = os.path.join(self.clean_image_dir, filename)
            noisy_image_path = os.path.join(self.noisy_image_dir, filename)
            clean_binary_image_path = os.path.join(self.clean_image_binary_mask_dir, filename)
            noisy_binary_image_path = os.path.join(self.noisy_image_binary_mask_dir, filename)
            noisy_text_binary_image_path = os.path.join(self.noist_text_binary_mask_dir, filename)

            cv2.imwrite(clean_image_path, clean_image)
            cv2.imwrite(clean_binary_image_path, clean_binary_image)
            cv2.imwrite(noisy_image_path, noisy_image)
            cv2.imwrite(noisy_binary_image_path, noisy_binary_image)
            cv2.imwrite(noisy_text_binary_image_path, noisy_text_binary_image)

            return self.df