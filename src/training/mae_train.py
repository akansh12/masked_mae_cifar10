import os
import math
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from einops import rearrange
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.models import MAE
from loss import mae_loss_function

def parse_args():
    parser = argparse.ArgumentParser(description='MAE Pre-training on CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Ratio of masked patches')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency of saving checkpoints')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, 'cifar10', 'mae-pretrain')
    os.makedirs(log_path, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup data
    train_transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    val_transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=val_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = MAE(
        img_size=32,
        patch_size=args.patch_size,
        in_chans=3,
        encoder_emb_dim=192,
        encoder_layers=12,
        encoder_heads=3,
        encoder_mlp_dim=768,
        decoder_layers=4,
        decoder_heads=3,
        decoder_mlp_dim=768,
        out_chans=3
    ).to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr * args.batch_size / 256,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            # Linear warmup
            return (epoch + 1) / (args.warmup_epochs + 1e-8)
        else:
            # Cosine decay
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lr_lambda,
        verbose=True
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_path)
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        with tqdm(total=len(train_loader)) as pbar:
            pbar.set_description(f"Epoch {epoch+1}/{args.epochs}")
            
            for step, (images, _) in enumerate(train_loader):
                global_step += 1
                images = images.to(device)
                
                # Forward pass
                pred_patches, mask = model(images)
                
                # Calculate loss
                loss = mae_loss_function(images, pred_patches, mask, patch_size=args.patch_size)
                loss = loss / args.accumulation_steps  # Normalize for gradient accumulation
                
                # Backward pass
                loss.backward()
                
                # Update weights (with gradient accumulation)
                if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Update metrics
                train_loss += loss.item() * args.accumulation_steps
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                pbar.update(1)
                
                # Log training loss
                writer.add_scalar('train/loss_step', loss.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('train/loss_epoch', avg_train_loss, epoch)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation and visualization
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Visualize reconstructions
                visualize_samples = []
                for i in range(min(16, len(val_dataset))):
                    img, _ = val_dataset[i]
                    visualize_samples.append(img)
                
                val_images = torch.stack(visualize_samples).to(device)
                pred_patches, mask = model(val_images)
                
                # Process images for visualization
                # Reshape predicted patches to image format
                pred_images = rearrange(
                    pred_patches, 
                    'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                    p1=args.patch_size, 
                    p2=args.patch_size,
                    h=32 // args.patch_size, 
                    w=32 // args.patch_size
                )
                
                # Create visualization grid
                # Original images with mask applied (showing only masked regions)
                masked_originals = val_images.clone()
                
                # Create a mask in image space
                mask_reshaped = rearrange(
                    mask, 
                    'b (h w) -> b 1 (h 1) (w 1)', 
                    h=32 // args.patch_size, 
                    w=32 // args.patch_size
                ).repeat(1, 1, args.patch_size, args.patch_size)
                mask_reshaped = mask_reshaped.repeat(1, 3, 1, 1)  # Repeat for RGB channels
                
                # Apply mask
                masked_originals = masked_originals * (1 - mask_reshaped)
                reconstructions = val_images * (1 - mask_reshaped) + pred_images * mask_reshaped
                
                # Concatenate for visualization: [masked input, reconstruction, ground truth]
                vis_images = torch.cat([masked_originals, reconstructions, val_images], dim=0)
                
                # Arrange in a grid
                grid = torchvision.utils.make_grid(
                    vis_images, 
                    nrow=8,
                    normalize=True,
                    value_range=(-1, 1)
                )
                
                writer.add_image('val/reconstructions', grid, epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f'mae_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        
        # Save best model
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_model_path = os.path.join(args.save_dir, 'mae_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f"Best model saved with loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()