"""
utils/checkpoint.py: Checkpointing utilities for QFLBackdoorAttacks
"""
import os
import torch

def save_checkpoint(model, optimizer, epoch, run_dir):
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'ckpt_epoch{epoch+1}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1
    }, ckpt_path)
    return ckpt_path

def load_checkpoint(path, model=None, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu')
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    return model, optimizer, epoch
