import pandas as pd
from nlpia.data.loaders import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.decomposition import PCA

pd.options.display.width = 120

sms = get_data('sms-spam')

index = ['sms{}{}'.format(i,'!'*j) for(i,j) in zip(range(len(sms)),sms.spam)]
sms.index = index
print(sms.head(7))

tfidf = TfidfVectorizer(tokenizer = casual_tokenize)
tfidf_docs =  tfidf.fit_transform(raw_documents = sms.text).toarray()

print(len(tfidf.vocabulary_))

tfidf_docs = pd.DataFrame(tfidf_docs)
tfidf_docs = tfidf_docs - tfidf_docs.mean() # this center your vectorized documents by substracting the mean (works like findind a centroid??)

print(tfidf_docs.shape)
print(sms.spam.sum())

pca = PCA(n_components = 16)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)

columns = ['topic{}'.format(i) for i in range(pca.n_components)]

pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns = columns)
print(pca_topic_vectors.round(3).head(6))
# assigning the words to the dimensions in our pca

column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(),tfidf.vocabulary_.keys())))
#print(termsi)
weights = pd.DataFrame(pca.components_, columns = terms,index = ['topic{}'.format(i)for i in range(16)])
pd.options.display.max_columns = 8
print(weights.head(4).round(3))

deals = weights['! ;) :) half off free crazy deal only $ 80 %'.split()].round(3)*100
print(deals)
print(deals.T.sum())
