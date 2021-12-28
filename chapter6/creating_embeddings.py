import nltk
from gensim.models.word2vec import Word2Vec

### let's select the corpus first

moby_dick = nltk.corpus.gutenberg.sents('melville-moby_dick.txt')
moby_dick = [sent for sent in moby_dick]

num_features = 300
min_word_count = 3
num_workers = 2
window_size = 6
subsambplin = 1e-3


model = Word2Vec(moby_dick, workers = num_features, vector_size=num_features,min_count=min_word_count, window=window_size, sample=subsambplin)

# this is to reduce the memory in the computer
model.init_sims(replace=True)
#step one
print('building vocabulary...')
model.build_vocab(moby_dick,progress_per = 10000)
#step two
print('training now...')
model.train(moby_dick,total_examples = model.corpus_count, epochs = 30, report_delay=1)
model_name = 'moby_dick_embeddings.bin'
print(model.wv['whale'])


model.save(model_name)
