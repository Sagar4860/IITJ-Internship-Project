# Multilingual Word Image Restoration & Synthetic Data Generation
This project aims to improve OCR robustness across diverse scripts and real-world noise conditions through synthetic data generation and restoration modeling. We simulate realistic degradations in multilingual word images and develop a restoration pipeline that significantly boosts OCR accuracy, especially for low-resource languages.


## 📁 Project Structure

```
├── demo_images/                         # Sample outputs for visualization
│   ├── noisy_images/                    # Noisy text images
│   └── synthetic_text_images/           # Corresponding clean text images

├── synthetic-data-generation/           # Text rendering + noise generation
│   └── synthetic_data_gen_and_noise.ipynb  # Combined notebook

├── metadata.csv                         # CSV with font, size, color, and noise info

├── word_restoration/                    # Main module: restoration + OCR + evaluation
│   ├── reconstruction/                  # Experiments on real-world test datasets
│   │   ├── IC17_test/                   # ICDAR 2017 dataset
│   │   ├── IGNCA_Dataset/Dataset/       # Sanskrit manuscripts
│   │   ├── test2/                       # Sample Images for every language
│   │   ├── configs/                     # Configuration files for experiments
│   │   ├── ocr/                         # Scripts to run OCR (Tesseract, TrOCR, EasyOCR)
│   │   ├── scripts/                     # Utilities and helpers
│   │   ├── 013.png ...                  # Sample inference images
│
│   ├── combined_visuals/                # Grid visualizations (clean, noisy, output, masks)
│   │   ├── combined_visual_grid_Arabic.png ...  # Visuals for every language
│
│   ├── IGNCA_augumentation.py           # Specific augmentation script for IGNCA
│   ├── augmentation.py                  # General augmentation logic
│   ├── finetune.py                      # Code to fine-tune OCR models
│   ├── hindi_dataset_gen.py             # Hindi-specific dataset generation
│   ├── ocr_analysis.ipynb               # OCR comparison, metrics, visualization
│   ├── view.py                          # Script to view combined outputs

├── README.md                            # You're reading it!
```

## Key Features
Multilingual Synthetic Data
Supports 15+ languages including Hindi, Tamil, Urdu, Chinese, German, Arabic, etc.

## Realistic Noise Simulation
Applies 40+ distortions like occlusions, blur, scratch, water damage, perspective, JPEG artifacts.

## Restoration Model (e.g., GSDM)
Learns to denoise and reconstruct word-level text images, improving OCR downstream.

## OCR Integration
Evaluates restored images using Tesseract, EasyOCR, TrOCR, with scripts to benchmark and analyze performance.

## Intermediate Representations
Generates ISM, CSM, and binary masks for better structural understanding and visualization.

## Applications
Benchmark restoration accuracy across different scripts and degradation types

Improve OCR accuracy in low-resource or noisy document scenarios

Fine-tune OCR models using restored outputs

## Visual Analysis
Combined grid visualizations for:

  -- Clean image
  
  -- Noisy input
  
  -- Restored output
  
  -- ISM (Intach segmentation Mask)
  
  -- CSM (Corrupted segmentation Mask)
  
  -- Binary mask

## Author
Sagar Premani
Intern, IIT Jodhpur – Word Image Restoration Project (2025)
