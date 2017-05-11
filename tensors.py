import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import sent_tokenize

from features import POSTokenizer
from run_model import load_features


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
    if n > 127:
        return 0
    elif n > 96:
        return n - 32 - 26
    else:
        return n - 32


def create_character_sentence_tensor(files, max_seq_length=50):
    texts = []  # list of text samples
    num_sentence_dict = {}
    for i, fname in enumerate(files):
        with open(fname) as f:
            text = f.read()
            sentences = sent_tokenize(text)
            num_sentence_dict[i] = len(sentences)
            for s in sentences:
                a = [char_to_i(c) for c in list((s.upper()))]
                texts.append(a)

    data = pad_sequences(texts, maxlen=max_seq_length)

    return np.asarray(data), num_sentence_dict


def create_character_tensor(files, tokenizer=None, max_seq_length=1000):
    texts = []

    for i, fname in enumerate(files):
        with open(fname) as f:
            text = f.read()
            a = [char_to_i(c) for c in list((text.upper()))]
            texts.append(a)

    b = pad_sequences(texts, maxlen=max_seq_length)
    return np.asarray(b)


create_stylo_tensor = load_features


def create_sentence_tensor(files, tokenizer=None, use_pos=False, max_seq_length=50, max_words=20000):
    texts = []  # list of text samples
    num_sentence_dict = {}
    for i, fname in enumerate(files):
        with open(fname) as f:
            text = f.read()
            sentences = sent_tokenize(text)
            num_sentence_dict[i] = len(sentences)
            for s in sentences:
                if use_pos:
                    t = POSTokenizer()
                    texts.append(' '.join(t(s)))
                else:
                    texts.append(s)

    print('Found %s texts.' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=max_seq_length)

    return data, word_index, tokenizer, num_sentence_dict


def create_tensor(files, tokenizer=None, use_pos=False, max_seq_length=1000, max_words=20000):
    texts = []  # list of text samples
    for fname in files:
        with open(fname) as f:
            if use_pos:
                t = POSTokenizer()
                texts.append(' '.join(t(f.read())))
            else:
                texts.append(f.read())

    print('Found %s texts.' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=max_seq_length)

    return data, word_index, tokenizer
