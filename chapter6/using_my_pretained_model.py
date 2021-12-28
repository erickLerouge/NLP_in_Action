from gensim.models.word2vec import Word2Vec

model_name = 'moby_dick_embeddings.bin'
model = Word2Vec.load(model_name)
print(model.wv.most_similar('whale'))
