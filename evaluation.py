import numpy as np
from run_model import encode_labels, prediction_results


def voting_test(model, test_tensored_in, test_orig_labels, test_num_sentences_dict):

    encoded_test_labels = encode_labels(test_orig_labels)

    predicted_labels = np.zeros(len(encoded_test_labels))
    predicted_maxxout_labels = np.zeros(len(encoded_test_labels))

    correct = 0.0
    total = 0.0
    tracker = 0
    for i in range(len(encoded_test_labels)):
        num_sentences = test_num_sentences_dict[i]
        test_sample = test_tensored_in[tracker:tracker+num_sentences, :]
        # print num_sentences, test_sample.shape
        predict_sample = model.predict(test_sample)

        predict_maxxout_sample = np.zeros(11)
        for s in predict_sample:
            j = np.argmax(s)
            predict_maxxout_sample[j] = predict_maxxout_sample[j] + 1

        predicted_maxxout_labels[i] = np.argmax(predict_maxxout_sample)

        voted_sample = np.sum(predict_sample, axis=0)

        predicted_labels[i] = np.argmax(voted_sample)

        if np.argmax(voted_sample) == encoded_test_labels[i]:
            correct += 1.0
        total += 1.0
        tracker += num_sentences

    print model.summary()
    print "Accuracy:",  str((correct/total))

    prediction_results(encoded_test_labels, predicted_labels)
    prediction_results(encoded_test_labels, predicted_maxxout_labels)
