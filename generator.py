import keras
import json
import numpy as np
import music21

class MelodyGenerator:

    def __init__(self, model_path: str, metadata_path: str):
        self.model = keras.models.load_model(model_path)
        
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
            self._mapping = metadata['mapping']
            self._sequence_length = metadata['sequence_length']
            self._num_classes = metadata['num_classes']

    def _sample_with_temperature(self, probabilities: np.ndarray, temperature: float) -> int:
        scaled_logits = np.log(probabilities) / temperature
        scaled_probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))

        return np.random.choice(range(len(probabilities)), p=scaled_probs)

    def _convert_to_stream(self, melody: list[str]) -> music21.stream.Part:
        part = music21.stream.Part()
        prev = None
        for symbol in melody:
            if symbol == '_':
                prev.quarterLength += 0.25
            elif symbol == 'r':
                prev = music21.note.Rest(quarterLength=0.25)
                part.append(prev)
            else:
                prev = music21.note.Note(int(symbol), quarterLength=0.25)
                part.append(prev)
        
        return part



    def generate_melody(self, seed: list[str], max_steps: int, max_sequence_length: int, temperature: float) -> music21.stream.Score:
        melody = seed
        seed = ['/'] * self._sequence_length + seed
        seed = [self._mapping[symbol] for symbol in seed]
        identity_matrix = np.eye(self._num_classes)

        for _ in range(max_steps):
            seed = seed[-max_sequence_length:]

            one_hot_seed = np.array([identity_matrix[symbol] for symbol in seed])
            input = np.expand_dims(one_hot_seed, axis=0)

            probabilities = self.model.predict(input)[0]

            output = self._sample_with_temperature(probabilities, temperature)
            seed.append(output)

            output_symbol = [key for key, value in self._mapping.items() if value == output][0]
            if output_symbol == '/':
                break

            melody.append(output_symbol)

        return self._convert_to_stream(melody)
    
    
if __name__ == '__main__':
    np.printoptions(threshold=np.inf)
    generator = MelodyGenerator('models/erk_model.keras', 'data/processed/erk_metadata.json')
    melody = generator.generate_melody(['69'], 1000, 64, 1.5)
    melody.show()




