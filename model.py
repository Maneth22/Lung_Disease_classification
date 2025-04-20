import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np

# Load Wave2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Custom Dataset
class LungSoundDataset(Dataset):
    def __init__(self, audio_files, labels, sample_rate=16000):
        self.audio_files = audio_files
        self.labels = labels
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        input_values = processor(waveform, sampling_rate=sr, return_tensors="pt").input_values
        
        # Extract Wave2Vec2 embeddings
        with torch.no_grad():
            features = wav2vec_model(input_values).last_hidden_state.squeeze(0)  # (T, C)
        
        return features, label

# Transformer Model for Classification
class TransformerLungClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_classes=3, num_layers=2):
        super(TransformerLungClassifier, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.transformer(x)  # (T, C)
        x = x.permute(1, 2, 0)  # (Batch, C, T)
        x = self.pooling(x).squeeze(-1)  # (Batch, C)
        x = self.fc(x)  # (Batch, num_classes)
        return x

# Training Setup
def train_model(train_loader, model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Load dataset (example)
audio_files = ["path/to/audio1.wav", "path/to/audio2.wav"]  # Replace with actual paths
labels = [0, 1]  # Example labels (0 = healthy, 1 = mild, 2 = severe disorder)
dataset = LungSoundDataset(audio_files, labels)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerLungClassifier(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(train_loader, model, criterion, optimizer)
