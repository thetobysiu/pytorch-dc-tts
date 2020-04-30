"""Data loader for Witcher 3 Dataset"""
import os
import re
import codecs
import numpy as np
import h5py

from torch.utils.data import Dataset

vocab = "PE abcdefghijklmnopqrstuvwxyz'.,"  # P: Padding, E: EOS.
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}


def text_normalize(text):
    text = text.lower()
    text = re.sub(f"[^{vocab}]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def read_metadata(metadata_file):
    fnames, text_lengths, texts = [], [], []
    transcript = os.path.join(metadata_file)
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    for line in lines:
        fname, _, text = line.strip().split("|")
        fnames.append(fname)
        text = text_normalize(text) + "E"  # E: EOS
        text = [char2idx[char] for char in text]
        text_lengths.append(len(text))
        texts.append(np.array(text, np.long))

    return fnames, text_lengths, texts


def get_test_data(sentences, max_n):
    normalized_sentences = [text_normalize(line).strip() + "E" for line in sentences]  # text normalization, E: EOS
    texts = np.zeros((len(normalized_sentences), max_n + 1), np.long)
    for i, sent in enumerate(normalized_sentences):
        texts[i, :len(sent)] = [char2idx[char] for char in sent]
    return texts


class Speech(Dataset):
    def __init__(self, keys, dir_name, file):
        self.keys = keys
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dir_name)
        self.fnames, self.text_lengths, self.texts = read_metadata(os.path.join(self.path, file))

    def slice(self, start, end):
        self.fnames = self.fnames[start:end]
        self.text_lengths = self.text_lengths[start:end]
        self.texts = self.texts[start:end]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        data = {}
        if 'texts' in self.keys:
            data['texts'] = self.texts[index]
        if 'mels' in self.keys:
            # (39, 80)
            while True:
                try:
                    with h5py.File(os.path.join(self.path, 'mels.hdf5'), "r") as mels:
                        data['mels'] = mels[self.fnames[index]][()]
                except KeyError:
                    continue
                break
        if 'mags' in self.keys:
            # (39, 80)
            while True:
                try:
                    with h5py.File(os.path.join(self.path, 'mags.hdf5'), "r") as mags:
                        data['mags'] = mags[self.fnames[index]][()]
                except KeyError:
                    continue
                break
        if 'mel_gates' in self.keys:
            data['mel_gates'] = np.ones(data['mels'].shape[0], dtype=np.int)  # TODO: because pre processing!
        if 'mag_gates' in self.keys:
            data['mag_gates'] = np.ones(data['mags'].shape[0], dtype=np.int)  # TODO: because pre processing!
        return data
