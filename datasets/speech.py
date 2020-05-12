"""Data loader for Witcher 3 Dataset"""
import os
import re
import codecs
import numpy as np

from hparams import HParams as hp
from torch.utils.data import Dataset
from utils import h5_loader

vocab = hp.vocab
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


def get_test_data(sentence):
    normalized_sentence = text_normalize(sentence) + "E"   # text normalization, E: EOS
    text = np.array([[char2idx[char] for char in normalized_sentence]], np.long)
    return text


class Speech(Dataset):
    keys = None
    path = None
    mels = None
    mags = None
    _fnames = None
    _text_lengths = None
    _texts = None
    vocab = hp.vocab

    def __init__(self, start, end):
        self.fnames = Speech._fnames[start:end]
        self.text_lengths = Speech._text_lengths[start:end]
        self.texts = Speech._texts[start:end]

    @classmethod
    def singleton(cls, name):
        if getattr(cls, name) is None:
            setattr(cls, name, h5_loader(os.path.join(cls.path, f'{name}.h5')))

    @classmethod
    def load(cls, keys, dir_name, file):
        cls.keys = keys
        cls.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dir_name)
        cls._fnames, cls._text_lengths, cls._texts = read_metadata(os.path.join(cls.path, file))

    @classmethod
    def get_script_length(cls):
        return len(cls._fnames)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        data = {}
        if 'texts' in self.keys:
            data['texts'] = self.texts[index]
        if 'mels' in self.keys:
            Speech.singleton('mels')
            data['mels'] = Speech.mels[self.fnames[index]][:]
        if 'mags' in self.keys:
            Speech.singleton('mags')
            data['mags'] = Speech.mags[self.fnames[index]][:]
        if 'mel_gates' in self.keys:
            data['mel_gates'] = np.ones(data['mels'].shape[0], dtype=np.int)  # TODO: because pre processing!
        if 'mag_gates' in self.keys:
            data['mag_gates'] = np.ones(data['mags'].shape[0], dtype=np.int)  # TODO: because pre processing!
        return data
