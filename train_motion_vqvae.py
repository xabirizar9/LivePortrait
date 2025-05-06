#!/usr/bin/env python
# coding: utf-8

"""
Training script for the MotionVQVAE model
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.modules.motion_vqvae import MotionVQVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Train MotionVQVAE")
    parser.add_argument("--config", type=str, default="src/config/models.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint_M", type=str,
                        default="pretrained_weights/liveportrait/base_models/motion_extractor.pth",
                        help="Path to motion extractor checkpoint")
    parser.add_argument("--output_dir", type=str, default="vqvae_checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")
    return parser.parse_args()


class DummyDataset(Dataset):
    """Dummy dataset for demonstration purposes"""
    def __init__(self, size=1000, image_size=256):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random RGB image
        return torch.randn(3, self.image_size, self.image_size)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    if not os.path.exists(path):
        return 0, float('inf')

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint['epoch'], checkpoint['loss']


def main():
    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Get parameters
    motion_extractor_params = config['model_params']['motion_extractor_params']

    # Create VQVAE parameters
    vqvae_params = {
        'code_num': 512,
        'code_dim': 512,
        'output_emb_width': 512,
        'down_t': 3,
        'stride_t': 2,
        'width': 512,
        'depth': 3,
        'dilation_growth_rate': 3,
        'activation': "relu",
        'apply_rotation_trick': True,
        'quantizer': "ema_reset"
    }

    # Create model
    print("Creating MotionVQVAE model...")
    model = MotionVQVAE(
        motion_extractor_params=motion_extractor_params,
        vqvae_params=vqvae_params
    )

    # Load motion extractor weights
    if os.path.exists(args.checkpoint_M):
        print(f"Loading motion extractor weights from {args.checkpoint_M}")
        state_dict = torch.load(args.checkpoint_M, map_location=lambda storage, loc: storage)
        model.load_motion_extractor(state_dict)
    else:
        print(f"Warning: Checkpoint {args.checkpoint_M} not found, using random weights")

    # Move model to device
    model = model.to(args.device)

    # Freeze motion extractor weights
    print("Freezing motion extractor weights...")
    for param in model.motion_extractor.parameters():
        param.requires_grad = False

    # Create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = DummyDataset(size=10000)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Training loop
    checkpoint_path = os.path.join(args.output_dir, "motion_vqvae.pt")
    start_epoch, best_loss = load_checkpoint(model, optimizer, checkpoint_path)

    print(f"Starting training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_mse_loss = 0.0
        running_commit_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for i, images in enumerate(progress_bar):
            # Move images to device
            images = images.to(args.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(images)

            # Extract losses
            commit_loss = output['commit_loss']

            # Calculate reconstruction loss using original motion info
            # and reconstructed motion info
            motion_info = model.motion_extractor(images)
            features = model._motion_info_to_features(motion_info)
            reconstructed_features = model._motion_info_to_features(output)

            mse_loss = F.mse_loss(features, reconstructed_features)

            # Combine losses
            loss = mse_loss + commit_loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update progress bar
            running_loss += loss.item()
            running_mse_loss += mse_loss.item()
            running_commit_loss += commit_loss.item()

            avg_loss = running_loss / (i + 1)
            avg_mse = running_mse_loss / (i + 1)
            avg_commit = running_commit_loss / (i + 1)
            perplexity = output['perplexity'].item()

            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'mse': f"{avg_mse:.4f}",
                'commit': f"{avg_commit:.4f}",
                'perplexity': f"{perplexity:.2f}"
            })

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)

            # Save epoch-specific checkpoint
            epoch_checkpoint_path = os.path.join(args.output_dir, f"motion_vqvae_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, epoch_checkpoint_path)

            # Update best model if needed
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(args.output_dir, "motion_vqvae_best.pt")
                save_checkpoint(model, optimizer, epoch, avg_loss, best_model_path)

    print("Training complete!")


if __name__ == "__main__":
    main()
