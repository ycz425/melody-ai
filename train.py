from preprocess import get_train_sequences, get_num_classes
import json
import keras

LOSS_FN = 'sparse_categorical_crossentropy'
LEARNING_RATE = 0.001
NUM_UNITS = [256]
EPOCHS = 100
BATCH_SIZE = 64
MODEL_SAVE_DIR = 'models'


def build_model(num_units: list[int], num_classes: int, loss_fn: str, learning_rate: float) -> keras.Model:
    input = keras.layers.Input(shape=(None, num_classes))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(input, output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_fn, metrics=['accuracy'])
    model.summary()

    return model


def train(data_file_name: str, num_units: list[int], learning_rate: float, loss_fn: str, batch_size: int, epochs: int, save_dir: str):
    inputs, targets = get_train_sequences(data_file_name)
    model = build_model(num_units, get_num_classes(data_file_name), loss_fn, learning_rate)

    model.fit(inputs, targets, batch_size=batch_size, epochs=epochs)
    model.save(save_dir)


def main():
    train('erk', NUM_UNITS, LEARNING_RATE, LOSS_FN, BATCH_SIZE, EPOCHS, MODEL_SAVE_DIR)


if __name__ == '__main__':
    main()
    