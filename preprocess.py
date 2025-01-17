import os
import music21
import json
import numpy as np
import xml.etree.ElementTree as ET


SAVE_DIR = 'data/processed'


def load(dataset_path: str) -> dict[str, music21.stream.Score]:
    scores = {}
    for dirpath, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename == 'c.xml':
                tree = ET.parse(os.path.join(dirpath, filename))
                root = tree.getroot()
                for harmony in root.findall('.//harmony'):
                    for frame in harmony.findall('.//frame'):
                        harmony.remove(frame)
                scores[dirpath] = (music21.converter.parse(ET.tostring(root, encoding='unicode'), format='musicxml'))         

    return scores


def durations_acceptable(song: music21.stream.Score, acceptable_durations: set[float]) -> bool:
    for note in song.flatten().getElementsByClass([music21.note.Note, music21.note.Rest]):
        if note.quarterLength not in acceptable_durations:
            return False
    return True


def encode_song(song: music21.stream.Score) -> list[str]:
    encoded_song = []
    prev = None

    for event in song.flatten().getElementsByClass(['Chord', 'Note', 'Rest']):
        if isinstance(event, music21.note.Note):
            symbol = str(event.pitch.midi)
            prev = 'note'
        elif isinstance(event, music21.harmony.ChordSymbol):
            symbol = event.figure.replace(' ', '_')
            if prev == 'chord':
                continue
            prev = 'chord'
        elif isinstance(event, music21.note.Rest):
            symbol = 'r'
            prev = 'note'

        encoded_song.append(symbol)

        if isinstance(event, (music21.note.Note, music21.note.Rest)):
            steps = int(event.quarterLength / 0.25)
            encoded_song.extend(['_'] * (steps - 1))
            prev = 'note'

    return encoded_song


def create_metadata(dir_name: str, encoded_songs: list[list[str]], dir_paths: list[str]) -> None:
    vocabulary = list(set([symbol for song in encoded_songs for symbol in song]))
    metadata = {}
    metadata['mapping'] = {symbol: index for index, symbol in enumerate(vocabulary)}
    metadata['dir_paths'] = dir_paths

    os.makedirs(os.path.join(SAVE_DIR, dir_name), exist_ok=True)
    with open(os.path.join(SAVE_DIR, f'{dir_name}/metadata.json'), 'w') as file:
        json.dump(metadata, file, indent=4)


def get_num_classes(dir_name: str) -> int:
    with open(os.path.join(SAVE_DIR, f'{dir_name}/metadata.json'), 'r') as file:
        metadata = json.load(file)

    return len(metadata['mapping'])


def preprocess(dataset_path: str, dir_name: str, acceptable_durations: set[float]) -> None:
    print('Loading songs...')
    songs = load(dataset_path)
    print(f'Loaded {len(songs)} songs.')

    result = []
    dir_paths = []
    for path in songs:
        if durations_acceptable(songs[path], acceptable_durations):
            result.append(['[START]'] + encode_song(songs[path]) + ['[END]'])
            dir_paths.append(path)

    os.makedirs(os.path.join(SAVE_DIR, dir_name), exist_ok=True)
    with open(os.path.join(SAVE_DIR, f'{dir_name}/encodings.txt'), 'w') as file:
        for song in result:
            file.write(' '.join(song) + '\n')

    create_metadata(dir_name, result, dir_paths)


def get_train_sequences(file_name: str) -> tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(SAVE_DIR, f'{file_name}.txt'), 'r') as file:
        songs = file.read().split()

    with open(os.path.join(SAVE_DIR, f'{file_name}_metadata.json'), 'r') as file:
        metadata = json.load(file)

    series = [metadata['mapping'][symbol] for symbol in songs]
    inputs = []
    targets = []

    for i in range(len(series) - metadata['sequence_length']):
        inputs.append(series[i:i + metadata['sequence_length']])
        targets.append(series[i + metadata['sequence_length']])

    identity_matrix = np.eye(metadata['num_classes'])
    inputs = np.array([[identity_matrix[symbol] for symbol in sequence] for sequence in inputs])
    targets = np.array(targets)

    return inputs, targets


def main():
    preprocess(
        dataset_path='data/raw/chord-melody-dataset-master',
        dir_name='data',
        acceptable_durations={0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0},
    )


if __name__ == '__main__':
    main()