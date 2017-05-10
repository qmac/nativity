from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer

import string
import enchant
import numpy as np


class POSTokenizer(object):

    def __call__(self, doc):
        tokenizer = RegexpTokenizer(r'\w+')
        return [tag[1] for tag in pos_tag(tokenizer.tokenize(doc))]


class StylometricFeatureExtractor(object):

    def extract(self, doc):
        spell_checker = enchant.Dict("en_US")
        sentences = sent_tokenize(str(doc))
        len_of_essay = len(sentences)

        sum_sent_lengths = 0.0
        sum_word_lengths = 0.0
        punc_count = 0.0
        num_mispelled_words = 0.0
        num_of_articles = 0.0

        passage = []
        for s in sentences:

            # with punctuation
            words_in_sentence = word_tokenize(s)

            for w in words_in_sentence:

                if w.startswith("'"):
                    w = w[1:]

                if w.endswith("\\n"):
                    w = w[:-2]

                if not w or w == "``" :
                    continue

                if w in list(string.punctuation):
                    punc_count += 1
                    continue

                if not spell_checker.check(w):
                    num_mispelled_words += 1

                if w == "the" or w == "a" or w == "an":
                    num_of_articles += 1

                passage.append(w)
                sum_word_lengths += len(w)

            sum_sent_lengths += len(words_in_sentence)

        lexical_variety = len(set(passage)) / float(len(passage))

        np_vec = np.matrix([num_mispelled_words,
                            num_of_articles,
                            len_of_essay,
                            punc_count,
                            sum_sent_lengths/len_of_essay,
                            sum_word_lengths/sum_sent_lengths,
                            lexical_variety])
        return np_vec


if __name__ == '__main__':
    ct = StylometricFeatureExtractor()
    print ct.extract("I fully agree to that statement because of the following reasons. \
First, an already successful person will accept a higher risk because he has a larger security if he fails. When the new thing produces loss he still has the gains of the successful working actions he did before. So he has a lower risk than a person who could lose everything by experimenting with new things. \
Secondly, the hunger for success grows with the success one has. A successful person knows there has to be more than the already achieved. The known old patterns will only provide the known old amount of success. But to increase it there have to be found new unknown and maybe also unfamiliar kinds of ways which mostly have a higher risk but also a higher possible gain than the conventional ones. If someone has a risky character he will not stop at the first success but look for more the whole life. \
The third and i think most important argument deals with experience with fear of the future. If a person were successful at least once in his or her life by taking a risk he or she has made an important step. By crossing the border in one's own mind once the next time it will be easier because you get a kind of imagination what ist waiting for you on the other side. Of course there is no possible calculation with all risks, there will always be some one did not think about. Despite of that the person knows how to deal with developments that can not be foreseen, so he or she has no or at least less fear of them. \
Thes arguments show, why success on a high level is not possible without risks at a high level and therefore i agree with the given statement. ")
