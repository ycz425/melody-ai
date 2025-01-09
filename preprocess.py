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


def durations_acceptable(song: music21.stream.Score, acceptable_durations: list[float]) -> bool:
    for note in song.flat.notesAndRests:
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

    for event in song.flat.notesAndRests:
        if isinstance(event, music21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, music21.note.Rest):
            symbol = 'r'

        steps = int(event.quarterLength / 0.25)
        encoded_song.append(symbol)
        encoded_song.extend(['_'] * (steps - 1))

    return ''.join(map(str, encoded_song))


def preprocess(dataset_path: str):
    print('Loading songs...')
    songs = load(dataset_path)
    print(f'Loaded {len(songs)} songs.')

    for song in songs:
        if durations_acceptable(song, ACCEPTABLE_DURATIONS):
            song = transpose_to_concert_pitch(song)
            song = encode_song(song)
