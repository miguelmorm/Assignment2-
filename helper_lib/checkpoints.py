# helper_lib/checkpoints.py
import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch:03d}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, checkpoint_path)
    print(f"✅ Checkpoint saved at: {checkpoint_path}")
