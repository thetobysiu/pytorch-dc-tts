#!/usr/bin/env python
"""Synthetize sentences into speech."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import sys
import argparse
from tqdm import *

import numpy as np
import torch

from models import Text2Mel, SSRN
from hparams import HParams as hp
from audio import save_to_wav
from utils import get_last_checkpoint_file_name, load_checkpoint, save_to_png
from datasets.speech import vocab, get_test_data

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", default='Geralt', help='dataset name')
args = parser.parse_args()

SENTENCES = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "Large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds."
]

torch.set_grad_enabled(False)

t2m_list = [f'logdir/Geralt-text2mel-256-64-0.005/step-{i:03}K.pth' for i in [80, 81]]

ssrn = SSRN().eval()
last_checkpoint_file_name = get_last_checkpoint_file_name(os.path.join(hp.logdir, '%s-ssrn' % args.dataset))
# last_checkpoint_file_name = 'logdir/%s-ssrn/step-005K.pth' % args.dataset
if last_checkpoint_file_name:
    print("loading ssrn checkpoint '%s'..." % last_checkpoint_file_name)
    load_checkpoint(last_checkpoint_file_name, ssrn, None)
else:
    print("ssrn not exits")
    sys.exit(1)

for t2m in t2m_list:
    filename = os.path.splitext(os.path.basename(t2m))[0]
    text2mel = Text2Mel(vocab).eval()
    load_checkpoint(t2m, text2mel, None)
    # text2mel = Text2Mel(vocab)
    # text2mel.load_state_dict(torch.load(t2m).state_dict())
    # text2mel = text2mel.eval()
    for i in range(len(SENTENCES)):
        sentences = [SENTENCES[i]]

        max_N = len(SENTENCES[i])
        L = torch.from_numpy(get_test_data(sentences, max_N))
        zeros = torch.from_numpy(np.zeros((1, hp.n_mels, 1), np.float32))
        Y = zeros
        A = None

        for t in tqdm(range(hp.max_T)):
            _, Y_t, A = text2mel(L, Y, monotonic_attention=True)
            Y = torch.cat((zeros, Y_t), -1)
            _, attention = torch.max(A[0, :, -1], 0)
            attention = attention.item()
            if L[0, attention] == vocab.index('E'):  # EOS
                break

        _, Z = ssrn(Y)

        Y = Y.cpu().detach().numpy()
        A = A.cpu().detach().numpy()
        Z = Z.cpu().detach().numpy()
        if not os.path.isdir(f'samples/{filename}'):
            os.mkdir(f'samples/{filename}')
        # save_to_png('samples/%d-att.png' % (i + 1), A[0, :, :])
        # save_to_png('samples/%d-mel.png' % (i + 1), Y[0, :, :])
        # save_to_png('samples/%d-mag.png' % (i + 1), Z[0, :, :])
        save_to_wav(Z[0, :, :].T, f'samples/{filename}/{i + 1}.wav')
