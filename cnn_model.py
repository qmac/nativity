'''
Taken/adapted from keras/examples
https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py
'''

from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.layers.merge import concatenate
from keras.models import Model


def make_cnn_model(embedding_layer, max_sequence_length=1000, use_dropout=True, use_stylo=False):
    # make a 1D convnet with global maxpooling
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = Dropout(0.2)(embedded_sequences) if use_dropout else embedded_sequences
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x) if use_dropout else x
    if use_stylo:
        stylo = Input(shape=(7,))
        x = concatenate([x, stylo])
    preds = Dense(11, activation='softmax')(x)

    model = Model([sequence_input, stylo], preds) if use_stylo else Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    print(model.summary())
    return model
