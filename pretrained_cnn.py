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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors

from run_model import load_files, encode_labels


WORD_VEC_FILE = '../GoogleNews-vectors-negative300.bin'
TEXT_DATA_DIR = '../nli-shared-task-2017/data/essays/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
NUM_LABELS = 11


def create_tensor(files, tokenizer=None):
    texts = []  # list of text samples
    for fname in files:
        with open(fname) as f:
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


# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

# embeddings_index = {}
# f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()

wvModel = KeyedVectors.load_word2vec_format(WORD_VEC_FILE, binary=True)

print('Found %s word vectors.' % len(wvModel.vocab))

# second, prepare text samples and their labels
print('Processing text dataset')

# Get the document files
train_files, train_labels, test_files, test_labels = load_files('train', 'dev')

# Encode labels
labels = to_categorical(np.array(encode_labels(train_labels)))

# Create data tensor
data, word_index, tokenizer = create_tensor(train_files)

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

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
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

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(NUM_LABELS, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))

# Evaluate
x_test, _, _ = create_tensor(test_files, tokenizer=tokenizer)
y_test = to_categorical(np.array(encode_labels(test_labels)))
print(model.evaluate(x_test, y_test))
# y_pred = model.predict(x_test)
# print(categorical_accuracy(y_test, y_pred))
