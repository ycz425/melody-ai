import os
import music21
import json
import numpy as np


SAVE_DIR = 'data/processed'


def load(dataset_path: str) -> list[music21.stream.Score]:
    scores = []
    for dirpath, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.endswith('.krn') or filename.endswith('.abc'):
                stream = music21.converter.parse(os.path.join(dirpath, filename))
                if isinstance(stream, music21.stream.Score):
                    scores.append(stream)
                elif isinstance(stream, music21.stream.Opus):
                    for score in stream:
                        scores.append(score)

    return scores


def durations_acceptable(song: music21.stream.Score, acceptable_durations: set[float]) -> bool:
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


def create_metadata_file(file_name: str, encoded_songs: str, sequence_length: int) -> None:
    metadata = {}

    vocabulary = list(set(encoded_songs.split()))
    mapping = {symbol: index for index, symbol in enumerate(vocabulary)}
    
    metadata['mapping'] = mapping
    metadata['sequence_length'] = sequence_length
    metadata['num_classes'] = len(mapping)

    with open(os.path.join(SAVE_DIR, f'{file_name}_metadata.json'), 'w') as file:
        json.dump(metadata, file)


def get_num_classes(data_file_name: str) -> int:
    with open(os.path.join(SAVE_DIR, f'{data_file_name}_metadata.json'), 'r') as file:
        metadata = json.load(file)

    return metadata['num_classes']


def preprocess(dataset_path: str, file_name: str, acceptable_durations: set[float], sequence_length: int) -> None:
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

    with open(os.path.join(SAVE_DIR, f'{file_name}.txt'), 'w') as file:
        file.write(result)

    create_metadata_file(file_name, result, sequence_length)


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
        dataset_path='data/raw/europa/deutschl/erk',
        file_name='erk',
        acceptable_durations={0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0},
        sequence_length=64
    )


if __name__ == '__main__':
    main()