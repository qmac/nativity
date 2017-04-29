from run_model import load_files
from gensim.models import Word2Vec

training_files, training_labels, test_files, test_labels = load_files('train', 'dev')
texts = [open(f).read() for f in training_files]
texts = [s.split() for s in texts]

model = Word2Vec(texts, size=300, window=5, min_count=1)
model.wv.save_word2vec_format('../trained_vectors.bin', binary=True)
