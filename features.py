from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer


class POSTokenizer(object):

    def __call__(self, doc):
        tokenizer = RegexpTokenizer(r'\w+')
        return [tag[1] for tag in pos_tag(tokenizer.tokenize(doc))]
