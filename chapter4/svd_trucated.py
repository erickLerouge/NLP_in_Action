import pandas as pd
from nlpia.data.loaders import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import numpy as np


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

svd = TruncatedSVD(n_components = 16, n_iter=100)


svd_topic_vectors = svd.fit_transform(tfidf_docs.values)
svd_topic_vectors = pd.DataFrame(svd_topic_vectors, columns=columns, index = index)

print(svd_topic_vectors.round(3).head(6))

#in order to see if the SVD model is good for classification we need to compute the cosine similarity between the members of the class.They should have inner class similar values. For the spam messages, we should see larger positive cosine similarity.

svd_topic_vectors = (svd_topic_vectors.T / np.linalg.norm(svd_topic_vectors, axis = 1)).T
print(svd_topic_vectors.iloc[:10].dot(svd_topic_vectors.iloc[:10].T).round(1))

