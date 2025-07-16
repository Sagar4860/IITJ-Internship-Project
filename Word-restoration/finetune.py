import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

class TrOCRDatasetFromCSV(Dataset):
    def __init__(self, csv_file, root_dir, processor):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir  # 'reconstruction/'
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        lang = row["lang_code"]
        filename = row["filename"]
        text = str(row["word"])

        image_path = os.path.join(self.root_dir, lang, "output", filename)
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

        return {"pixel_values": pixel_values, "labels": labels}


from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader
import torch

# Initialize processor & model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# Dataset & Loader
dataset = TrOCRDatasetFromCSV("data.csv", "reconstruction", processor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
num_epochs = 3
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader)}")

# Save the fine-tuned model
model.save_pretrained("trocr_finetuned_multilingual")
processor.save_pretrained("trocr_finetuned_multilingual")
