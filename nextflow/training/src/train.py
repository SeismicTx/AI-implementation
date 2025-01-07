#!/usr/bin/env python3

import argparse
import json
import torch
import pandas as pd
from model import AntibodyPropertyPredictor
from torch.utils.data import DataLoader


def train_model(model: torch.nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int = 10) -> Dict:
    """Train the model and return metrics"""
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    metrics = {
        'train_loss': [],
        'val_loss': [],
    }

    for epoch in range(epochs):
        # Training loop implementation
        pass

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--metrics', required=True)
    args = parser.parse_args()

    # Load data and train model
    data = pd.read_csv(args.input)
    model = AntibodyPropertyPredictor()
    metrics = train_model(model, data)

    # Save model and metrics
    torch.save(model.state_dict(), args.output)
    with open(args.metrics, 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    main()
