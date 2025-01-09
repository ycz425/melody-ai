import os
import music21
import json
import numpy as np


KERN_DATASET_PATH = 'data/raw/europa/deutschl/test'
ACCEPTABLE_DURATIONS = {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0}
SAVE_DIR = 'data/processed'
SEQUENCE_LENGTH = 64


def load(dataset_path: str) -> list[music21.stream.Score]:
    songs = []
    for dirpath, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.endswith('.krn'):
                songs.append(music21.converter.parse(os.path.join(dirpath, filename)))

    return songs


def durations_acceptable(song: music21.stream.Score, acceptable_durations: list[float]) -> bool:
    for note in song.flatten().notesAndRests:
        if note.quarterLength not in acceptable_durations:
            return False
    return True
        

def transpose_to_concert_pitch(song: music21.stream.Score) -> music21.stream.Score:
    parts = song.getElementsByClass(music21.stream.Part)
    part0_measures = parts[0].getElementsByClass(music21.stream.Measure)
    key = part0_measures[0][4]

    if not isinstance(key, music21.key.Key):
        key = song.analyze('key')

    if key.mode == 'major':
        interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch('C'))
    elif key.mode == 'minor':
        interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch('A'))

    return song.transpose(interval)


def encode_song(song: music21.stream.Score) -> str:
    encoded_song = []

    for event in song.flatten().notesAndRests:
        if isinstance(event, music21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, music21.note.Rest):
            symbol = 'r'

        steps = int(event.quarterLength / 0.25)
        encoded_song.append(symbol)
        encoded_song.extend(['_'] * (steps - 1))

    return ' '.join(map(str, encoded_song))


def create_mapping(encoded_songs: str, file_name: str, save_dir: str) -> None:
    vocabulary = list(set(encoded_songs.split()))
    mapping = {symbol: index for index, symbol in enumerate(vocabulary)}

    with open(os.path.join(save_dir, f'{file_name}_mapping.json'), 'w') as file:
        json.dump(mapping, file)


def preprocess(dataset_path: str, file_name: str, save_dir: str, acceptable_durations: set[float], sequence_length: int) -> None:
    print('Loading songs...')
    songs = load(dataset_path)
    print(f'Loaded {len(songs)} songs.')

    result = []
    for song in songs:
        if durations_acceptable(song, acceptable_durations):
            song = transpose_to_concert_pitch(song)
            song = encode_song(song)

            result.append(song)
            result.extend(['/'] * sequence_length)

    result = ' '.join(result)

    with open(os.path.join(save_dir, f'{file_name}.txt'), 'w') as file:
        file.write(result)

    create_mapping(result, file_name, save_dir)


def get_train_sequences(file_name: str, save_dir: str, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(save_dir, f'{file_name}.txt'), 'r') as file:
        songs = file.read().split()

    with open(os.path.join(save_dir, f'{file_name}_mapping.json'), 'r') as file:
        mapping = json.load(file)

    series = [mapping[symbol] for symbol in songs]
    inputs = []
    targets = []

    for i in range(len(series) - sequence_length):
        inputs.append(series[i:i + sequence_length])
        targets.append(series[i + sequence_length])
    
    inputs = np.array([[np.eye(len(mapping))[symbol] for symbol in sequence] for sequence in inputs])
    targets = np.array(targets)

    return inputs, targets


def main():
    preprocess(KERN_DATASET_PATH, 'test', SAVE_DIR, ACCEPTABLE_DURATIONS, SEQUENCE_LENGTH)
    inputs, targets = get_train_sequences('test', SAVE_DIR, SEQUENCE_LENGTH)


if __name__ == '__main__':
    main()