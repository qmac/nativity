# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from gensim.models.keyedvectors import KeyedVectors

from run_model import load_files, encode_labels
from word_cnn import create_tensor

WORD_VEC_FILE = '../trained_vectors.bin'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300

class LstmModel(object):

    def fit(self, X_train, y_train, X_test, y_test, top_words=50000, embed_layer=None):


        (n, l) = y_train.shape
        # fix random seed for reproducibility
        numpy.random.seed(7)

        # truncate and pad input sequences
        # max_review_length = 500
        # X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
        # X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

        # create the model
        embedding_vecor_length = 32
        model = Sequential()
        if embed_layer is None:
            model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        else:
            model.add(embed_layer)
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dropout(0.2))
        model.add(Dense(l, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


        print(model.summary())
        model.fit(X_train, y_train, nb_epoch=10, batch_size=128, validation_data=(X_test, y_test))

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))


def setup():
    pass

if __name__ == '__main__':

    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000


    # Get the document files
    train_files, train_labels, test_files, test_labels = load_files('train', 'dev')

    # Encode labels and then make them one hots
    encoded_training_labels = encode_labels(train_labels)
    encoded_test_labels = encode_labels(test_labels)

    X_train, train_wi, train_tokenizer = create_tensor(train_files)

    X_test, test_wi, test_tokenizer = create_tensor(test_files)



    WORD_VEC_FILE = '../trained_vectors.bin'
    wvModel = KeyedVectors.load_word2vec_format(WORD_VEC_FILE, binary=True)
    num_words = min(MAX_NB_WORDS, len(train_wi))
    embedding_matrix = numpy.zeros((num_words, EMBEDDING_DIM))
    unknown_count = 0
    for word, i in train_wi.items():
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


    one_hot_y_train =  numpy.eye(11)[encoded_training_labels]
    one_hot_y_test =   numpy.eye(11)[encoded_test_labels]

    print "-------- X Train -------------", X_train.shape
    print X_train
    print "-------- Y Train -------------", one_hot_y_train.shape
    print one_hot_y_train

    print "-------- X Test --------------", X_test.shape
    print X_test

    print "-------- Y Test --------------", one_hot_y_train.shape
    print one_hot_y_test

    print "------------------------------"

    lstm_model = LstmModel()


    # lstm_model.fit(X_train, y_train, X_test, y_test)
    lstm_model.fit(X_train, one_hot_y_train, X_test, one_hot_y_test, top_words=num_words, embed_layer=embedding_layer)
