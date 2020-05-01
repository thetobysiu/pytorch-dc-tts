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

print('packing...')
with h5py.File(os.path.join(voice_path, 'data.hdf5'), "w") as data:
    for mel in tqdm(mels_list):
        data.create_dataset(f'mels/{os.path.splitext(os.path.basename(mel))[0]}',
                            chunks=True, fletcher32=True, dtype='float32', data=np.load(mel))
    for mag in tqdm(mags_list):
        data.create_dataset(f'mags/{os.path.splitext(os.path.basename(mag))[0]}',
                            chunks=True, fletcher32=True, dtype='float32', data=np.load(mag))
