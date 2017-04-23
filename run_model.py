import csv
import os
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC, SVC

SCRIPT_DIR = '/Users/quinnmac/Documents/NLP/Final Project/nli-shared-task-2017/scripts/'
CLASS_LABELS = ['ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR']  # valid labels


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


def load_ngrams(file_list, labels, vectorizer=None, fit=False):
    # convert label strings to integers
    labels_encoded = [CLASS_LABELS.index(label) for label in labels]
    if vectorizer is None:
        vectorizer = CountVectorizer(input="filename")  # create a new one
        doc_term_matrix = vectorizer.fit_transform(file_list)
    elif fit:
        doc_term_matrix = vectorizer.fit_transform(file_list)
    else:
        doc_term_matrix = vectorizer.transform(file_list)

    print("Created a document-term matrix with %d rows and %d columns."
          % (doc_term_matrix.shape[0], doc_term_matrix.shape[1]))

    return doc_term_matrix.astype(float), labels_encoded, vectorizer


def pretty_print_cm(cm, class_labels):
    row_format = "{:>5}" * (len(class_labels) + 1)
    print(row_format.format("", *class_labels))
    for l1, row in zip(class_labels, cm):
        print(row_format.format(l1, *row))


if __name__ == '__main__':
    training_partition_name = 'train'
    test_partition_name = 'dev'
    preprocessor = 'tokenized'

    vectorizer = CountVectorizer(input='filename', ngram_range=(1, 2), min_df=1)

    #
    # Load the training and test features and labels
    #
    training_files, training_labels, test_files, test_labels = load_files(training_partition_name, test_partition_name)
    training_matrix, encoded_training_labels, vectorizer = load_ngrams(training_files,
                                                                       training_labels,
                                                                       vectorizer,
                                                                       fit=True)
    test_matrix, encoded_test_labels,  _ = load_ngrams(test_files, test_labels, vectorizer)

    #
    # Run the classifier
    #

    # Normalize frequencies to unit length
    # transformer = Normalizer()
    transformer = TfidfTransformer()
    training_matrix = transformer.fit_transform(training_matrix)
    testing_matrix = transformer.fit_transform(test_matrix)

    # Train the model
    # Check the scikit-learn documentation for other models
    print("Training the classifier...")
    clf = LinearSVC()
    # clf = SVC(cache_size=7000)
    # clf = KNeighborsClassifier()
    # clf = RandomForestClassifier(n_estimators=200)
    clf.fit(training_matrix, encoded_training_labels)  # Linear kernel SVM
    predicted = clf.predict(testing_matrix)

    # Run cross val on training data
    cv_score = cross_val_score(clf, training_matrix, encoded_training_labels, cv=10).mean()
    print("Cross validation score: {}".format(cv_score))

    #
    # Display classification results
    #
    if -1 not in encoded_test_labels:
        print("\nConfusion Matrix:\n")
        cm = metrics.confusion_matrix(encoded_test_labels, predicted).tolist()
        pretty_print_cm(cm, CLASS_LABELS)
        print("\nClassification Results:\n")
        print(metrics.classification_report(encoded_test_labels, predicted, target_names=CLASS_LABELS))
    else:
        print("The test set labels aren't known, cannot print accuracy report.")