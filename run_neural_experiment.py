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

import numpy as np
from keras.layers import Embedding
from keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors

from run_model import load_files, encode_labels, load_features
from evaluation import voting_test
from tensors import (expand_labels,
                     create_character_tensor,
                     create_word_tensor)
from cnn_model import make_cnn_model
from lstm_model import make_lstm_model


# WORD_VEC_FILE = '../GoogleNews-vectors-negative300.bin'
WORD_VEC_FILE = '../trained_vectors.bin'
TEXT_DATA_DIR = '../nli-shared-task-2017/data/essays/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
NUM_LABELS = 11
NUM_EPOCHS = 10
PRETRAINED_EMBEDDINGS = False
USE_POS_TAGS = False
USE_DROPOUT = True
USE_STYLO = False
SAVE_MODEL = False
USE_CNN = False  # otherwise, lstm
SENTENCE_LEVEL = True  # otherwise, doc level
USE_WORDS = True  # otherwise, character level


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

    # Load x's
    if USE_WORDS:
        data, word_index, tokenizer, train_sentence_dict = create_word_tensor(train_files,
                                                         sentence_level=SENTENCE_LEVEL,
                                                         max_seq_length=MAX_SEQUENCE_LENGTH,
                                                         max_words=MAX_NB_WORDS)
        x_test, _, _, test_sentence_dict = create_word_tensor(test_files,
                                          sentence_level=SENTENCE_LEVEL,
                                          tokenizer=tokenizer,
                                          max_seq_length=MAX_SEQUENCE_LENGTH,
                                          max_words=MAX_NB_WORDS)
    else:
        data, train_sentence_dict = create_character_tensor(train_files,
                                                         sentence_level=SENTENCE_LEVEL,
                                                         max_seq_length=MAX_SEQUENCE_LENGTH)
        x_test, test_sentence_dict = create_character_tensor(test_files,
                                          sentence_level=SENTENCE_LEVEL,
                                          max_seq_length=MAX_SEQUENCE_LENGTH)
        word_index = np.arange(72)

    # Load y's
    labels = to_categorical(np.array(encode_labels(train_labels)))
    y_test = to_categorical(np.array(encode_labels(test_labels)))
    if SENTENCE_LEVEL:
        labels = expand_labels(labels, train_sentence_dict)
        y_test = expand_labels(y_test, test_sentence_dict)

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
        wvModel = None
    else:
        print('Preparing embedding layer')

        embedding_layer = Embedding(num_words,
                                    EMBEDDING_DIM,
                                    input_length=MAX_SEQUENCE_LENGTH)

    if USE_STYLO:
        stylo_tensor = load_features(train_files)
        stylo_tensor = stylo_tensor[indices]
        stylo_train = stylo_tensor[:-num_validation_samples]
        stylo_val = stylo_tensor[-num_validation_samples:]
        stylo_test = load_features(test_files)
        x_train = [x_train, stylo_train]
        x_val = [x_val, stylo_val]
        x_test = [x_test, stylo_test]

    print('Training model.')

    if USE_CNN:
        model = make_cnn_model(embedding_layer, use_dropout=USE_DROPOUT, max_sequence_length=MAX_SEQUENCE_LENGTH, use_stylo=USE_STYLO)
    else:
        model = make_lstm_model(embedding_layer, use_dropout=USE_DROPOUT)

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=NUM_EPOCHS,
              validation_data=(x_val, y_val))

    # Evaluate
    print(model.evaluate(x_test, y_test))

    if SAVE_MODEL:
        model.save('char_cnn_model.hdf')

    if SENTENCE_LEVEL:
        voting_test(model, x_test, test_labels, test_sentence_dict)
