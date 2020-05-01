"""Utility methods."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import glob
import torch
import h5py
from skimage.io import imsave


def get_last_checkpoint_file_name(logdir):
    """Returns the last checkpoint file name in the given log dir path."""
    checkpoints = glob.glob(os.path.join(logdir, '*.pth'))
    checkpoints.sort()
    if len(checkpoints) == 0:
        return None
    return checkpoints[-1]


def load_checkpoint(checkpoint_file_name, model, optimizer):
    """Loads the checkpoint into the given model and optimizer."""
    checkpoint = torch.load(checkpoint_file_name)
    model.load_state_dict(checkpoint['state_dict'])
    model.float()
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    del checkpoint
    print("loaded checkpoint epoch=%d step=%d" % (start_epoch, global_step))
    return start_epoch, global_step


def h5_loader(filepath, mode='r', swmr=False, driver=None):
    return h5py.File(filepath, mode=mode, libver='latest', swmr=swmr, driver=driver)


def save_checkpoint(logdir, epoch, global_step, model, optimizer):
    """Saves the training state into the given log dir path."""
    checkpoint_file_name = os.path.join(logdir, f'step-{global_step:06d}.pth')
    print(f"saving the checkpoint file '{checkpoint_file_name}'...")
    checkpoint = {
        'epoch': epoch + 1,
        'global_step': global_step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_file_name)
    del checkpoint


def save_to_png(file_name, array):
    """Save the given numpy array as a PNG file."""
    # from skimage._shared._warnings import expected_warnings
    # with expected_warnings(['precision']):
    imsave(file_name, array)
