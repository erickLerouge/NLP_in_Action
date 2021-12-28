from nlpia.data.loaders import get_data
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDiA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

ldia = LDiA(n_components = 16, learning_method='batch')

np.random.seed(42)

pd.options.display.width = 120

sms = get_data('sms-spam')

index = ['sms{}{}'.format(i,'!'*j) for(i,j) in zip(range(len(sms)),sms.spam)]
sms.index = index
print(sms.head(7))

counter = CountVectorizer(tokenizer=casual_tokenize)
bow_docs = pd.DataFrame(counter.fit_transform(raw_documents = sms.text).toarray(), index = index)

column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(),counter.vocabulary_.keys())))
bow_docs.columns = terms

#column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(),tfidf.vocabulary_.keys())))

print(sms.loc['sms0'].text)
print(bow_docs.loc['sms0'][bow_docs.loc['sms0']>0].head())

ldia = ldia.fit(bow_docs)
print(ldia.components_.shape)

## creatng the name of the colums
columns = ['topic{}'.format(i) for i in range(16)]
#print(columns)
# creatng the name of the colums
columns = ['topic{}'.format(i) for i in range(16)]
components = pd.DataFrame(ldia.components_.T, index = terms, columns = columns)

print(components.round(2).head(4))

print(components.topic3.sort_values(ascending=False)[:10])

# getting the topic vectors

ldia16_topic_vectors = ldia.transform(bow_docs)
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors,index= index, columns = columns)
print(ldia16_topic_vectors.round(2).head())
X_train, X_test, y_train, y_test = train_test_split(ldia16_topic_vectors,sms.spam, test_size = 0.5, random_state = 271828)

lda = LDA(n_components = 1)
lda = lda.fit(X_train,y_train)
sms['ldia16_spam'] = lda.predict(ldia16_topic_vectors)
# lda.score() return the mean accuracy on the fiven test data and labels
print(round(float(lda.score(X_test, y_test)),2))

