import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

class AntibodyDataset(Dataset):
    def __init__(self, sequences_df: pd.DataFrame, experimental_df: pd.DataFrame):
        self.data = sequences_df.merge(experimental_df, on='Sample ID')
        
        # Create amino acid vocabulary
        self.aa_vocab = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        
        # Standardize experimental features
        self.scaler = StandardScaler()
        self.exp_features = ['KD (nM)', 'Tm1', 'Tm2', '% Monomer']
        self.exp_data = self.scaler.fit_transform(self.data[self.exp_features])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Convert sequences to numerical arrays
        vh_tensor = torch.tensor([self.aa_vocab[aa] for aa in row['VH_seq']], dtype=torch.long)
        vl_tensor = torch.tensor([self.aa_vocab[aa] for aa in row['VL_seq']], dtype=torch.long)
        
        # Get experimental features
        exp_tensor = torch.tensor(self.exp_data[idx], dtype=torch.float)
        
        return {
            'vh': vh_tensor,
            'vl': vl_tensor,
            'exp_features': exp_tensor,
            'sample_id': row['Sample ID']
        }

class AntibodyPropertyPredictor(nn.Module):
    def __init__(self, vocab_size: int = 20, embedding_dim: int = 32, 
                 hidden_dim: int = 64, num_exp_features: int = 4):
        super().__init__()
        
        # Sequence processing
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vh_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.vl_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Experimental feature processing
        self.exp_linear = nn.Linear(num_exp_features, hidden_dim)
        
        # Combined processing
        self.combine_layer = nn.Linear(hidden_dim * 4 + hidden_dim, hidden_dim)
        
        # Output layers
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, vh, vl, exp_features):
        # Process VH sequence
        vh_emb = self.embedding(vh)
        vh_out, (vh_hidden, _) = self.vh_lstm(vh_emb)
        vh_feat = torch.cat((vh_hidden[-2,:,:], vh_hidden[-1,:,:]), dim=1)
        
        # Process VL sequence
        vl_emb = self.embedding(vl)
        vl_out, (vl_hidden, _) = self.vl_lstm(vl_emb)
        vl_feat = torch.cat((vl_hidden[-2,:,:], vl_hidden[-1,:,:]), dim=1)
        
        # Process experimental features
        exp_feat = self.exp_linear(exp_features)
        
        # Combine all features
        combined = torch.cat([vh_feat, vl_feat, exp_feat], dim=1)
        hidden = F.relu(self.combine_layer(combined))
        
        # Generate prediction
        output = self.output_layer(hidden)
        return output

def train_model(model: nn.Module, train_loader: DataLoader, 
                num_epochs: int = 100, learning_rate: float = 0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch['vh'], batch['vl'], batch['exp_features'])
            loss = criterion(output, batch['target'].unsqueeze(1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return train_losses

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Tuple[List[float], List[float]]:
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch['vh'], batch['vl'], batch['exp_features'])
            predictions.extend(output.squeeze().tolist())
            actuals.extend(batch['target'].tolist())
    
    return predictions, actuals