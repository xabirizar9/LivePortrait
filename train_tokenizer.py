import torch
import numpy as np
import pandas as pd
import cv2
import imageio
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch.nn.functional as F
import torch.distributed

from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader

from src.modules.motion_extractor import MotionExtractor
from src.live_portrait_wrapper import LivePortraitWrapper
from src.modules.vqvae import VQVae


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.video_dir = self.data_path / "train"

        self.video_list = pd.read_csv(self.data_path / "videos_by_timestamp.csv")

        self.video_paths = [self.video_dir / f"{video_id}.mp4" for video_id in self.video_list['original_video_id'].unique()]


    def prepare_videos(self, imgs) -> torch.Tensor:
        """ construct the input as standard
        imgs: NxBxHxWx3, uint8
        """

        N, H, W, C = imgs.shape

        _imgs = imgs.reshape(N, H, W, C, 1)

        y = _imgs.astype(np.float32) / 255.
        y = np.clip(y, 0, 1)  # clip to 0~1
        y = torch.from_numpy(y).permute(0, 4, 3, 1, 2)  # TxHxWx3x1 -> Tx1x3xHxW

        return y

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        video = imageio.get_reader(video_path)
        frames = np.array([frame for frame in video])

        output = self.prepare_videos(frames)

        return output

    def __len__(self):
        return len(self.video_paths)


class VQVAEModule(pl.LightningModule):
    def __init__(self, nfeats=72, code_num=512, code_dim=512, output_emb_width=512,
                 down_t=3, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
                 activation="relu", apply_rotation_trick=False, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.m_extr = MotionExtractor()
        self.m_extr.load_pretrained(init_path="pretrained_weights/liveportrait/base_models/motion_extractor.pth")

        # Freeze MotionExtractor parameters to prevent training
        for param in self.m_extr.parameters():
            param.requires_grad = False

        # Explicitly set model to eval mode to ensure inference behavior
        self.m_extr.eval()

        # Verify parameters are frozen
        trainable_params = sum(p.numel() for p in self.m_extr.parameters() if p.requires_grad)
        if trainable_params > 0:
            print(f"WARNING: MotionExtractor has {trainable_params} trainable parameters!")
        else:
            print("MotionExtractor is properly frozen - no trainable parameters.")

        self.vqvae = VQVae(
            nfeats=nfeats,
            code_num=code_num,
            code_dim=code_dim,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            apply_rotation_trick=apply_rotation_trick
        )

        self.lr = lr

    def training_step(self, batch, batch_idx):
        # Ensure MotionExtractor is in eval mode
        self.m_extr.eval()

        # Extract keypoints for each frame in the batch (only working for batch size 1)
        with torch.no_grad():
            kp_vid = torch.stack([self.m_extr(image)['kp'].squeeze(0) for image in batch[0]])
        kp_vid = kp_vid.unsqueeze(0)
        # Forward pass through VQVAE
        reconstr, commit_loss, perplexity = self.vqvae(kp_vid)

        # Calculate reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstr, kp_vid)

        # Total loss
        total_loss = recon_loss + commit_loss

        # Determine if we're using multiple GPUs
        sync_dist = True  # Always sync metrics in distributed training

        # Log metrics with proper sync_dist setting
        self.log('train_total_loss', total_loss, prog_bar=True, sync_dist=sync_dist, rank_zero_only=True)
        self.log('train_recon_loss', recon_loss, sync_dist=sync_dist, rank_zero_only=True)
        self.log('train_commit_loss', commit_loss, sync_dist=sync_dist, rank_zero_only=True)
        self.log('train_perplexity', perplexity, sync_dist=sync_dist, rank_zero_only=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Ensure MotionExtractor is in eval mode
        self.m_extr.eval()

        # Extract keypoints for each frame in the batch
        with torch.no_grad():
            kp_vid = torch.stack([self.m_extr(image)['kp'].squeeze(0) for image in batch[0]])
        kp_vid = kp_vid.unsqueeze(0)

        # Forward pass through VQVAE
        reconstr, commit_loss, perplexity = self.vqvae(kp_vid)

        # Calculate reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstr, kp_vid)

        # Total loss
        total_loss = recon_loss + commit_loss

        # Determine if we're using multiple GPUs
        sync_dist = True  # Always sync metrics in distributed training

        # Log metrics with proper sync_dist setting
        self.log('val_total_loss', total_loss, prog_bar=True, sync_dist=sync_dist, rank_zero_only=True)
        self.log('val_recon_loss', recon_loss, sync_dist=sync_dist, rank_zero_only=True)
        self.log('val_commit_loss', commit_loss, sync_dist=sync_dist, rank_zero_only=True)
        self.log('val_perplexity', perplexity, sync_dist=sync_dist, rank_zero_only=True)

        return total_loss

    def configure_optimizers(self):
        # Only optimize VQVAE parameters, not MotionExtractor
        optimizer = torch.optim.Adam(self.vqvae.parameters(), lr=self.lr)
        return optimizer


class StepTrackingCallback(pl.Callback):
    """Custom callback to track and report training steps."""

    def __init__(self, save_steps=500):
        super().__init__()
        self.save_steps = save_steps
        self.last_saved_step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Get the current global step
        step = trainer.global_step

        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"Training step: {step}")

        # Save model manually at specific intervals if ModelCheckpoint didn't
        if step % self.save_steps == 0 and step > self.last_saved_step:
            self.last_saved_step = step

            # Only save on rank 0 (main process)
            if trainer.is_global_zero:
                # Create path for manual checkpoint
                checkpoint_dir = Path(trainer.checkpoint_callback.dirpath)
                checkpoint_dir.mkdir(exist_ok=True, parents=True)

                # Get current loss
                current_loss = trainer.callback_metrics.get("train_total_loss", 0.0)

                # Create both a PyTorch state dict (smaller file) and a full checkpoint
                model_path = checkpoint_dir / f"vqvae-step-{step:06d}-loss-{current_loss:.4f}.pth"
                ckpt_path = checkpoint_dir / f"vqvae-step-{step:06d}-loss-{current_loss:.4f}.ckpt"

                # Save the model state dict (just the model parameters)
                torch.save(pl_module.vqvae.state_dict(), model_path)
                print(f"Saved model state dict at step {step} to {model_path}")

                # Save the full training state (model, optimizer, etc.) using PyTorch Lightning
                trainer.save_checkpoint(ckpt_path)
                print(f"Saved full training state at step {step} to {ckpt_path}")


def main(args):
    # Initialize wandb only on the main process
    if torch.cuda.is_available() and args.num_gpus > 1:
        is_main_process = (os.environ.get('LOCAL_RANK', '0') == '0')
    else:
        is_main_process = True

    # Only initialize wandb on the main process
    if is_main_process:
        wandb.init(project="vqvae-tokenizer", name=args.run_name, config=vars(args))

    # Set up data
    train_dataset = Dataset(args.data_path)
    print(f"Loaded {len(train_dataset)} videos")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    # Set up validation data if provided
    val_loader = None
    if args.val_data_path:
        val_dataset = Dataset(args.val_data_path)
        print(f"Loaded {len(val_dataset)} validation videos")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    # Set up model
    model = VQVAEModule(
        nfeats=72,
        code_num=512,
        code_dim=512,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        apply_rotation_trick=False,
        lr=args.learning_rate
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(args.output_path / 'checkpoints'),
        filename='vqvae-{epoch:02d}-step-{step}-loss-{train_total_loss:.4f}',
        save_top_k=3,
        save_last=True,  # Always save the last model
        every_n_train_steps=args.save_every_n_steps,  # Save every N training steps
        monitor='train_total_loss',
        mode='min'
    )

    # Add a validation checkpoint if validation data is provided
    callbacks = [checkpoint_callback]
    if val_loader:
        val_checkpoint_callback = ModelCheckpoint(
            dirpath=str(args.output_path / 'val_checkpoints'),
            filename='vqvae-val-{epoch:02d}-loss-{val_total_loss:.4f}',
            save_top_k=3,
            monitor='val_total_loss',
            mode='min'
        )
        callbacks.append(val_checkpoint_callback)

    # Setup step tracking callback
    step_tracking_callback = StepTrackingCallback(save_steps=args.save_every_n_steps)
    callbacks.append(step_tracking_callback)

    # Set up wandb logger only on main process
    logger = None
    if is_main_process:
        logger = WandbLogger(project="liveportrait-tokenizer", name=args.run_name, log_model="all")
        # Log hyperparameters
        logger.log_hyperparams(vars(args))

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.num_gpus if torch.cuda.is_available() else 1,
        strategy='ddp' if torch.cuda.is_available() and args.num_gpus > 1 else None,
        log_every_n_steps=10,
        sync_batchnorm=True if torch.cuda.is_available() and args.num_gpus > 1 else False
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Save final model
    final_model_path = str(args.output_path / 'vqvae_final.pth')
    if trainer.global_rank == 0:  # Only save on the main process
        model_state = model.vqvae.state_dict()
        torch.save(model_state, final_model_path)
        print(f"Saved final model to {final_model_path}")

    # Finish wandb run only on main process
    if is_main_process:
        wandb.finish()


if __name__ == "__main__":
    args = ArgumentParser()

    args.add_argument("--data_path", type=Path, default="dataset", required=True)
    args.add_argument("--val_data_path", type=str, default=None)
    args.add_argument("--output_path", type=Path, default="models", required=True)
    args.add_argument("--run_name", type=str, default="vqvae-training")
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--num_workers", type=int, default=1)
    args.add_argument("--max_epochs", type=int, default=100)
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--num_gpus", type=int, default=1)
    args.add_argument("--save_every_n_steps", type=int, default=500)

    args = args.parse_args()

    main(args)

# python train_tokenizer.py \
# --data_path dataset \
# --output_path models \
# --batch_size 1 \
# --num_workers 4 \
# --max_epochs 1 \
# --learning_rate 3e-4
