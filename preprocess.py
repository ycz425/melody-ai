import os
import music21
from music21 import environment


KERN_DATASET_PATH = 'data/europa/deutschl/test'
ACCEPTABLE_DURATIONS = {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0}


def load(dataset_path: str) -> list[music21.stream.Score]:
    songs = []
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.endswith('.krn'):
                songs.append(music21.converter.parse(os.path.join(dirpath, filename), format='kern'))

    return songs


def preprocess(dataset_path: str):
    print('Loading songs...')
    songs = load(dataset_path)
    print(f'Loaded {len(songs)} songs.')



