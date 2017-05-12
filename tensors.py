import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import sent_tokenize

from features import POSTokenizer


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


def create_character_tensor(files, sentence_level=False, max_seq_length=1000):
    texts = []
    num_sentence_dict = {}
    for i, fname in enumerate(files):
        with open(fname) as f:
            text = f.read()
            if sentence_level:
                sentences = sent_tokenize(text)
                num_sentence_dict[i] = len(sentences)
                for s in sentences:
                    a = [char_to_i(c) for c in list((s.upper()))]
                    texts.append(a)
            else:
                a = [char_to_i(c) for c in list((text.upper()))]
                texts.append(a)

    b = pad_sequences(texts, maxlen=max_seq_length)
    return np.asarray(b), num_sentence_dict


def create_word_tensor(files, tokenizer=None, sentence_level=False, use_pos=False, max_seq_length=1000, max_words=20000):
    texts = []  # list of text samples
    num_sentence_dict = {}
    for i, fname in enumerate(files):
        with open(fname) as f:
            text = f.read()
            if sentence_level:
                sentences = sent_tokenize(text)
                num_sentence_dict[i] = len(sentences)
                for s in sentences:
                    if use_pos:
                        t = POSTokenizer()
                        texts.append(' '.join(t(s)))
                    else:
                        texts.append(s)
            else:
                if use_pos:
                    t = POSTokenizer()
                    texts.append(' '.join(t(text)))
                else:
                    texts.append(text)

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
