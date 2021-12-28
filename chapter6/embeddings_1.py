#from nlpia.data.loaders import get_data
from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format('/Users/velazquezerick/Documents/NLP_in_action/GoogleNews-vectors-negative300.bin',binary=True)
#word_vectors = get_data('word2vec')

# this is equal to make the add opperations
print(word_vectors.most_similar(positive=['cooking','potatoes'],topn=5))
print(word_vectors.most_similar(positive=['france','germany'], topn=5))

#not related terms:

print(word_vectors.doesnt_match("potatoes milk cake computer".split()))
### operation with king+women - man

print(word_vectors.most_similar(positive=['king','woman'], negative=['man'], topn=5))
