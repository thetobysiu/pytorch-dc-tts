import os
from audio import preprocess

datasets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
voice_path = os.path.join(datasets_path, 'LJ')

preprocess(voice_path)
print("done")
