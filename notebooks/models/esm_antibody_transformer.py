import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import esm
from typing import List, Tuple
from tqdm import tqdm

class ESMAntibodyDataset(Dataset):
    def __init__(self, merged_experimental_df: str):
        """
        Args:
            merged_experimental_df: Path to CSV with columns for VH/VL sequences and experimental data
        """
        self.data = pd.read_csv(merged_experimental_df)
        
        # Load ESM-2 model and tokenizer
        # We'll use the smallest ESM model (8M parameters) for performance, 
        # but for best results the larger versions should be used
        self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # Set to eval mode since we're only using for embeddings
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Prepare sequences for ESM
        vh_data = [("vh", row['sequences_hc_sequence'])]
        vl_data = [("vl", row['sequences_lc_sequence'])]
        
        # Get embeddings for VH and VL
        with torch.no_grad():
            # Process VH
            _, _, vh_tokens = self.batch_converter(vh_data)
            vh_results = self.model(vh_tokens, repr_layers=[6])
            vh_embeddings = vh_results["representations"][6]  # Use last layer
            
            # Process VL
            _, _, vl_tokens = self.batch_converter(vl_data)
            vl_results = self.model(vl_tokens, repr_layers=[6])
            vl_embeddings = vl_results["representations"][6]
        
        # Extract target variables
        targets = torch.tensor([
            row['binding_affinity_kd'],
            row['thermostability_tm1_celsius'],
            row['asec_monomerpct']
        ], dtype=torch.float)
        
        return (vh_embeddings, vl_embeddings), targets

class ESMAntibodyTransformer(nn.Module):
    def __init__(self, esm_dim=320, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        """
        Args:
            esm_dim: Dimension of ESM embeddings (320 for ESM-2 8M)
            d_model: Internal transformer dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Project ESM embeddings to transformer dimension
        self.vh_projection = nn.Linear(esm_dim, d_model)
        self.vl_projection = nn.Linear(esm_dim, d_model)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(d_model * 2, 256)  # *2 because we concatenate VH and VL
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)  # 3 outputs: KD, Tm1, POI
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        vh_embed, vl_embed = x
        
        # Project ESM embeddings
        vh = self.vh_projection(vh_embed.squeeze(0))
        vl = self.vl_projection(vl_embed.squeeze(0))
        
        # Pass through transformer
        vh_encoded = self.transformer(vh)
        vl_encoded = self.transformer(vl)
        
        # Pool sequence dimension using attention-weighted mean
        vh_pooled = vh_encoded.mean(dim=1)  # [batch_size, d_model]
        vl_pooled = vl_encoded.mean(dim=1)  # [batch_size, d_model]
        
        # Concatenate VH and VL features
        combined = torch.cat([vh_pooled, vl_pooled], dim=1)
        
        # Final MLP layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
):
    """Training loop with validation"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for (vh_embed, vl_embed), targets in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            vh_embed = vh_embed.to(device)
            vl_embed = vl_embed.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model((vh_embed, vl_embed))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for (vh_embed, vl_embed), targets in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                vh_embed = vh_embed.to(device)
                vl_embed = vl_embed.to(device)
                targets = targets.to(device)
                
                outputs = model((vh_embed, vl_embed))
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_esm_model.pth')