'''
Taken/adapted from keras/examples
https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py


This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import numpy as np
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM
from keras.models import Model
from keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import sent_tokenize

from run_model import load_files, encode_labels
from features import POSTokenizer
from evaluation import voting_test


WORD_VEC_FILE = '../trained_vectors.bin'
TEXT_DATA_DIR = '../nli-shared-task-2017/data/essays/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 72
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
NUM_LABELS = 11
PRETRAINED_EMBEDDINGS = False
USE_POS_TAGS = False
USE_DROPOUT = False


def expand_labels(labels, num_sentence_dict):
    expanded_labels = []
    for i in range(len(labels)):
        num_sentences = num_sentence_dict[i]
        expanded_labels.extend(num_sentences * [labels[i]])

    return np.array(expanded_labels)


def char_to_i(c):
    c = c.upper()
    n = ord(c)
    if n < 32:
        if n == 9 or n == 10:
            return n + 60
    if n > 96:
        return n - 32 - 26
    else:
        return n - 32


def create_character_tensor(files, tokenizer=None):
    texts = []

    for i, fname in enumerate(files):
        with open(fname) as f:
            text = f.read()
            a = [char_to_i(c) for c in list((text.upper()))]
            texts.append(a)

    b = pad_sequences(texts, maxlen=MAX_SEQUENCE_LENGTH)
    

    # data = [ np.eye(71)[b[i]] for i in range(len(b))]        

    return np.asarray(b)

def create_sentence_tensor(files, tokenizer=None):
    texts = []  # list of text samples
    num_sentence_dict = {}
    for i, fname in enumerate(files):
        with open(fname) as f:
            text = f.read()
            sentences = sent_tokenize(text)
            num_sentence_dict[i] = len(sentences)
            for s in sentences:
                if USE_POS_TAGS:
                    t = POSTokenizer()
                    texts.append(' '.join(t(s)))
                else:
                    texts.append(s)

    print('Found %s texts.' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return data, word_index, tokenizer, num_sentence_dict


def create_tensor(files, tokenizer=None):
    texts = []  # list of text samples
    for fname in files:
        with open(fname) as f:
            if USE_POS_TAGS:
                t = POSTokenizer()
                texts.append(' '.join(t(f.read())))
            else:
                texts.append(f.read())

    print('Found %s texts.' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return data, word_index, tokenizer


if __name__ == '__main__':
    # first, build index mapping words in the embeddings set
    # to their embedding vector

    if PRETRAINED_EMBEDDINGS:
        print('Indexing word vectors.')

        wvModel = KeyedVectors.load_word2vec_format(WORD_VEC_FILE, binary=True)

        print('Found %s word vectors.' % len(wvModel.vocab))

    # second, prepare text samples and their labels
    print('Processing text dataset')

    # Get the document files
    train_files, train_labels, test_files, test_labels = load_files('train', 'dev')

    # Encode labels
    labels = to_categorical(np.array(encode_labels(train_labels)))

    # Create data tensor
    # data, word_index, tokenizer, num_sentence_dict = create_sentence_tensor(train_files)
    data = create_character_tensor(train_files)
    word_index = np.arange(71)

    # labels = expand_labels(labels, num_sentence_dict)

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    num_words = min(MAX_NB_WORDS, len(word_index)+1)

    if PRETRAINED_EMBEDDINGS:
        print('Preparing embedding matrix.')

        # prepare embedding matrix
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        unknown_count = 0
        for word, i in word_index.items():
            if i >= MAX_NB_WORDS:
                continue
            if word in wvModel:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = wvModel[word]
            else:
                unknown_count += 1

        print('Found %d unknown words.' % unknown_count)

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(num_words,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
    else:
        print('Preparing embedding layer')

        embedding_layer = Embedding(num_words,
                                    EMBEDDING_DIM,
                                    input_length=MAX_SEQUENCE_LENGTH)

    print('Training model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = Dropout(0.2)(embedded_sequences) if USE_DROPOUT else embedded_sequences
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x) if USE_DROPOUT else x
    preds = Dense(NUM_LABELS, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    print(model.summary())
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=15,
              validation_data=(x_val, y_val))


    # Evaluate
    # x_test, _, _, test_sentence_dict = create_sentence_tensor(test_files, tokenizer=tokenizer)

    # y_test = expand_labels(to_categorical(np.array(encode_labels(test_labels))), test_sentence_dict)
    x_test = create_character_tensor(test_files)
    y_test = np.eye(11)[encode_labels(test_labels)]
    print(model.evaluate(x_test, y_test))

    model.save('char_cnn_model.hdf')
    # voting_test(model, x_test, test_labels, test_sentence_dict)
