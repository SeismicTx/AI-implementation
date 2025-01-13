import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

class AntibodyDataset(Dataset):
    def __init__(self, merged_experimental_df: pd.DataFrame):
        self.data = pd.read_csv(merged_experimental_df)

        # Create amino acid vocabulary
        self.aa_vocab = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Convert sequences to numerical arrays
        vh_tensor = torch.tensor([self.aa_vocab[aa] for aa in row['sequences_hc_sequence']], dtype=torch.long)
        vl_tensor = torch.tensor([self.aa_vocab[aa] for aa in row['sequences_lc_sequence']], dtype=torch.long)

        # Extract target variables
        target_columns = ['binding_affinity_kd', 'thermostability_tm1_celsius', 'asec_monomerpct']
        targets = row.loc[target_columns].astype(float).values

        # Pad the VL to have the same length as the VH
        return pad_sequence([vh_tensor, vl_tensor]), torch.tensor(targets)

class AntibodyTransformer(nn.Module):
    def __init__(self, vocab_size=20, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()

        # Embedding layer for amino acid sequences
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout
        )

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.pos_encoder,
            num_layers=num_layers
        )

        # Output layers
        self.fc1 = nn.Linear(d_model * 2, 256)  # *2 because we have VH and VL
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)  # 3 outputs: aggregation, KD, Tm1

        self.dropout = nn.Dropout(dropout)

        self.double()

    def forward(self, x):
        # x shape: [batch_size, 2, seq_length]
        batch_size = x.shape[0]

        # Process VH and VL sequences separately
        vh = x[:, 0, :]  # [batch_size, seq_length]
        vl = x[:, 1, :]  # [batch_size, seq_length]

        # Embed sequences
        vh_embedded = self.embedding(vh).transpose(0, 1)  # [seq_length, batch_size, d_model]
        vl_embedded = self.embedding(vl).transpose(0, 1)  # [seq_length, batch_size, d_model]

        # Pass through transformer
        vh_encoded = self.transformer(vh_embedded)  # [seq_length, batch_size, d_model]
        vl_encoded = self.transformer(vl_embedded)  # [seq_length, batch_size, d_model]

        # Pool sequence dimension
        vh_pooled = vh_encoded.mean(dim=0)  # [batch_size, d_model]
        vl_pooled = vl_encoded.mean(dim=0)  # [batch_size, d_model]

        # Concatenate VH and VL features
        combined = torch.cat([vh_pooled, vl_pooled], dim=1)

        # Final MLP layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4):
    """Training loop with validation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)

                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
