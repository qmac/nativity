import csv
import os
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC

from features import POSTokenizer, StylometricFeatureExtractor

SCRIPT_DIR = '../nli-shared-task-2017/scripts/'
CLASS_LABELS = ['ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR']  # valid labels


# Loads files from training data
def load_files(train_partition, test_partition, preprocessor='tokenized'):
    train_labels_path = "{script_dir}/../data/labels/{train}/labels.{train}.csv".format(train=train_partition, script_dir=SCRIPT_DIR)
    test_labels_path = "{script_dir}/../data/labels/{test}/labels.{test}.csv".format(test=test_partition, script_dir=SCRIPT_DIR)

    #
    #  Read labels files. If feature files provided, `training_files` and `test_files` below will be ignored
    #
    with open(train_labels_path) as train_labels_f, open(test_labels_path) as test_labels_f:
        essay_path_train = '{script_dir}/../data/essays/{train}/{preproc}'.format(script_dir=SCRIPT_DIR, train=train_partition, preproc=preprocessor)
        essay_path_test = '{script_dir}/../data/essays/{test}/{preproc}'.format(script_dir=SCRIPT_DIR, test=test_partition, preproc=preprocessor)

        training_files, training_labels = zip(*[(os.path.join(essay_path_train, row['test_taker_id'] + '.txt'), row['L1'])
                                                for row in csv.DictReader(train_labels_f)])

        test_files, test_labels = zip(*[(os.path.join(essay_path_test, row['test_taker_id'] + '.txt'), row['L1'])
                                        for row in csv.DictReader(test_labels_f)])

    return training_files, training_labels, test_files, test_labels


# Turn labels from "ITA" to a number
def encode_labels(labels):
    return [CLASS_LABELS.index(label) for label in labels]


# Loads n-grams given the vectorizer of from the files in file list
def load_ngrams(file_list, labels, vectorizer=None, fit=False):
    # convert label strings to integers
    if vectorizer is None:
        vectorizer = CountVectorizer(input="filename")  # create a new one
        doc_term_matrix = vectorizer.fit_transform(file_list)
    elif fit:
        doc_term_matrix = vectorizer.fit_transform(file_list)
    else:
        doc_term_matrix = vectorizer.transform(file_list)

    print("Created a document-term matrix with %d rows and %d columns."
          % (doc_term_matrix.shape[0], doc_term_matrix.shape[1]))

    return doc_term_matrix.astype(float), vectorizer


# Loads stylometric features
def load_features(file_list):
    vectorizer = StylometricFeatureExtractor()

    file_name = file_list[0]
    f = open(file_name, 'r')
    matrix = vectorizer.extract(f.readlines())
    f.close()

    for i in range(1, len(file_list)):
        file_name = file_list[i]
        f = open(file_name, 'r')
        matrix = np.append(matrix, vectorizer.extract(f.readlines()), axis=0)
        f.close()

    return scale(matrix)


def pretty_print_cm(cm, class_labels):
    row_format = "{:>5}" * (len(class_labels) + 1)
    print(row_format.format("", *class_labels))
    for l1, row in zip(class_labels, cm):
        print(row_format.format(l1, *row))


def prediction_results(expected, predicted):
    if -1 not in expected:
        print("\nConfusion Matrix:\n")
        cm = metrics.confusion_matrix(expected, predicted).tolist()
        pretty_print_cm(cm, CLASS_LABELS)
        print("\nClassification Results:\n")
        print(metrics.classification_report(expected, predicted, target_names=CLASS_LABELS))
    else:
        print("The test set labels aren't known, cannot print accuracy report.")


if __name__ == '__main__':
    training_partition_name = 'train'
    test_partition_name = 'dev'
    preprocessor = 'tokenized'

    # Get the document files
    training_files, training_labels, test_files, test_labels = load_files(training_partition_name, test_partition_name)

    # Encode labels
    encoded_training_labels = encode_labels(training_labels)
    encoded_test_labels = encode_labels(test_labels)

    # Create training and testing data
    vectorizer = CountVectorizer(input='filename', analyzer='char', ngram_range=(2, 7), min_df=1)
    # Uncomment for POS tag data
    # vectorizer = CountVectorizer(input='filename', tokenizer=POSTokenizer(), ngram_range=(1, 7))
    training_matrix, vectorizer = load_ngrams(training_files,
                                              training_labels,
                                              vectorizer,
                                              fit=True)
    testing_matrix,  _ = load_ngrams(test_files, test_labels, vectorizer)

    # Normalize frequencies to unit length
    transformer = TfidfTransformer()
    training_matrix = transformer.fit_transform(training_matrix)
    testing_matrix = transformer.fit_transform(testing_matrix)

    # Uncomment for stylometric analysis
    # training_matrix = load_features(training_files)
    # testing_matrix = load_features(test_files)

    # Train the model
    print("Training the classifier...")
    clf = LinearSVC()
    clf.fit(training_matrix, encoded_training_labels)  # Linear kernel SVM
    predicted = clf.predict(testing_matrix)

    # Display classification results
    prediction_results(encoded_test_labels, predicted)
    print(accuracy_score(encoded_test_labels, predicted))

    # Run cross val on training data
    cv_score = cross_val_score(clf, training_matrix, encoded_training_labels, cv=10).mean()
    print("Cross validation score: {}".format(cv_score))
