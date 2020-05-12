#!/usr/bin/env python
"""Synthetize sentences into speech."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import sys
import glob
# import argparse
from tqdm import *

import numpy as np
import torch

from models import Text2Mel, SSRN
from hparams import HParams as hp
from audio import save_to_wav
from utils import get_last_checkpoint_file_name, load_checkpoint, save_to_png
from datasets.speech import vocab, get_test_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--dataset", default='Geralt', help='dataset name')
# args = parser.parse_args()

SENTENCES = [
    "A striped tomcat sleeping on a sun-warmed stack of wood, shuddered, raised his round head, pulled back his ears, hissed and bolted off into the nettles.",
    "Three-year- old Dragomir, fisherman Trigla’s son, who was sitting on the hut’s threshold doing his best to make dirtier an already dirty shirt,",
    "started to scream as he fixed his tearful eyes on the passing rider.",
    # "The witcher rode slowly, without trying to overtake the hay- cart obstructing the road.",
    # "A laden donkey trotted behind him, stretching its neck and constantly pulling the cord tied to the witcher’s pommel tight.",
    # "In addition to the usual bags, the long- eared animal was lugging a large shape, wrapped in a saddlecloth, on its back.",
    # "The gray- white flanks of the ass were covered with black streaks of dried blood.",
    # "The cart finally turned down a side street leading to a granary and harbor from which a sea breeze blew, carrying the stink of tar and ox’s urine.",
    # "Geralt picked up his pace. He didn’t react to the muffled cry of the woman selling vegetables who was staring at the bony, ",
    # "taloned paw sticking out beneath the horse blanket, bobbing up and down in time with the donkey’s trot.",
    # "He didn’t look round at the crowd gathering behind him and rippling with excitement.",
    # "There were, as usual, many carts in front of the alderman’s house.",
    # "Geralt jumped wooden barrier. The crowd following him formed a semi- circle around the donkey."
    # # "The birch canoe slid on the smooth planks.",
    # "Glue the sheet to the dark blue background.",
    # # "It's easy to tell the depth of a well.",
    # "These days a chicken leg is a rare dish.",
    # "Rice is often served in round bowls.",
    # "The juice of lemons makes fine punch.",
    # "The box was thrown beside the parked truck.",
    # "The hogs were fed chopped corn and garbage.",
    # "Four hours of steady work faced us.",
    # "Large size in stockings is hard to sell.",
    # "The boy was there when the sun rose.",
    # "A rod is used to catch pink salmon.",
    # "The source of the huge river is the clear spring.",
    # "Kick the ball straight and follow through.",
    # "Help the woman get back to her feet.",
    # "A pot of tea helps to pass the evening.",
    # "Smoky fires lack flame and heat.",
    # "The soft cushion broke the man's fall. But you can't be serious, can you? Right, This is the sentence. So thank me, please.",
    # "The salt breeze came across from the sea.",
    # "The girl at the booth sold fifty bonds."
]

torch.set_grad_enabled(False)

t2m_list = glob.glob(f'logdir/LJ-lj_fixed.csv-256-0.005-64-text2mel/step-*.pth')
# t2m_list = ['logdir/Keira-Keira_all.csv-512-0.005-16-text2mel/step-010000.pth']
# t2m_list = ['logdir/Geralt-Geralt_s5_no_a.csv-256-0.005-32-text2mel/step-093500.pth']
#             'logdir/Geralt-512-0.005-24-text2mel/step-073500 (copy).pth',
#             'logdir/Geralt-512-0.005-24-text2mel/step-063000 (copy).pth']

ssrn = SSRN().to(device).eval()
print("loading ssrn...")
load_checkpoint('trained/ssrn/lj/step-140K.pth', ssrn, None)
# last_checkpoint_file_name = get_last_checkpoint_file_name(os.path.join(hp.logdir, '%s-ssrn' % args.dataset))
# last_checkpoint_file_name = 'logdir/%s-ssrn/step-005K.pth' % args.dataset
# if last_checkpoint_file_name:
#     print("loading ssrn checkpoint '%s'..." % last_checkpoint_file_name)
#     load_checkpoint(last_checkpoint_file_name, ssrn, None)
# else:
#     print("ssrn not exits")
#     sys.exit(1)
if not os.path.isdir(f'samples'):
    os.mkdir(f'samples')

for t2m in t2m_list:
    filename = os.path.splitext(os.path.basename(t2m))[0]
    folder = os.path.split(os.path.split(t2m)[0])[-1]
    text2mel = Text2Mel(vocab).to(device).eval()
    print("loading text2mel...")
    load_checkpoint(t2m, text2mel, None)
    # text2mel = Text2Mel(vocab)
    # text2mel.load_state_dict(torch.load(t2m).state_dict())
    # text2mel = text2mel.eval()
    for sentence in SENTENCES:
        with torch.no_grad():
            L = torch.from_numpy(get_test_data(sentence)).to(device)
            zeros = torch.from_numpy(np.zeros((1, hp.n_mels, 1), np.float32)).to(device)
            Y = zeros
            # A = None

            while True:
                _, Y_t, A = text2mel(L, Y, monotonic_attention=True)
                Y = torch.cat((zeros, Y_t), -1)
                _, attention = torch.max(A[0, :, -1], 0)
                attention = attention.item()
                if L[0, attention] == vocab.index('E'):  # EOS
                    print(f'{sentence} ok!')
                    break

            _, Z = ssrn(Y)

            # Y = Y.cpu().detach().numpy()
            # A = A.cpu().detach().numpy()
            Z = Z.cpu().detach().numpy()
        if not os.path.isdir(f'samples/{folder}'):
            os.mkdir(f'samples/{folder}')
        if not os.path.isdir(f'samples/{folder}/{filename}'):
            os.mkdir(f'samples/{folder}/{filename}')
        # save_to_png('samples/%d-att.png' % (i + 1), A[0, :, :])
        # save_to_png('samples/%d-mel.png' % (i + 1), Y[0, :, :])
        # save_to_png('samples/%d-mag.png' % (i + 1), Z[0, :, :])
        save_to_wav(Z[0, :, :].T, f'samples/{folder}/{filename}/{sentence}.wav')
