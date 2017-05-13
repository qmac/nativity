# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

numpy.random.seed(7)

# Makes LSTM model
def make_lstm_model(embedding_layer, use_dropout=True):
    # create the model
    model = Sequential()
    model.add(embedding_layer)
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(LSTM(32))
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(Dense(11, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
