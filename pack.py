# SIU KING WAI
import os
import h5py
import glob
import numpy as np
from tqdm import tqdm

datasets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
voice_path = os.path.join(datasets_path, 'LJ')

wavs_path = os.path.join(voice_path, 'wavs')
mels_path = os.path.join(voice_path, 'mels')
mags_path = os.path.join(voice_path, 'mags')

mels_list = glob.glob(f'{mels_path}/*.npy')
mags_list = glob.glob(f'{mags_path}/*.npy')

print('packing mels...')
with h5py.File(os.path.join(voice_path, 'mels.hdf5'), "w") as mels:
    for mel in tqdm(mels_list):
        mels.create_dataset(os.path.splitext(os.path.basename(mel))[0], data=np.load(mel))
print('packing mags...')
with h5py.File(os.path.join(voice_path, 'mags.hdf5'), "w") as mags:
    for mag in tqdm(mags_list):
        mags.create_dataset(os.path.splitext(os.path.basename(mag))[0], data=np.load(mag))
