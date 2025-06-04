# Synthetic Text Image Generation with Noise (IITJ Internship)

This project aims to generate synthetic text images using diverse fonts and apply various real-world noise patterns to simulate document degradations. The goal is to build a rich dataset for training robust OCR or document enhancement models.

---

## 📁 Project Structure

```
├── demo images/                         # Sample outputs (clean + noisy)
│   ├── noisy_images/                    # Noisy samples
│   └── synthetic_text_images/           # Clean text images
│
├── synthetic-data-generation/          # Scripts and notebooks
│   └── synthetic_data_gen_and_noise.ipynb  # Main code notebook (merged version)
│
├── metadata.csv                         # CSV logging font, size, color, noise per image
├── README.md                            # You're reading it!

---

## 🔧 Features

- Generates text images using random:
  - Fonts (from font-family directory)
  - Colors and font sizes
  - English words from vocabulary
- Applies **one random noise type per image** from over 40 real-world distortions (blur, compression, pixelation, etc.)
- Logs metadata in CSV (image name, font, size, color, noise, etc.)

---

## ✍️ Author

**Sagar Premani**  
IITJ Internship 2025
