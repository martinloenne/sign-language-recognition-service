from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
from tensorflow.keras import optimizers


class LSTMModel:
    def __init__(self, name, lstm_layers, dense_layers, layer_size, input_shape, output_size,
                 lstm_activation, dense_activation, output_activation, dropout):
        self.name = name

        model = Sequential()
        model.add(Input(shape=input_shape))

        # LSTM layers
        if lstm_layers < 1:
            raise Exception("Model must have at least one LSTM layer.")

        for i in range(lstm_layers - 1):
            model.add(LSTM(layer_size, activation=lstm_activation, return_sequences=True))

        model.add(LSTM(layer_size, activation=lstm_activation))

        # Dense layers
        for i in range(dense_layers):
            model.add(Dense(layer_size, activation=dense_activation))
            model.add(Dropout(dropout))

        # Output layer
        model.add(Dense(output_size))
        model.add(Activation(output_activation))

        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

        self.src = model
