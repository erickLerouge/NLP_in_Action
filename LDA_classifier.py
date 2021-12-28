#Linear discriminad analysis
import pandas as pd
from nlpia.data.loaders import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.preprocessing import MinMaxScaler
from pugnlp.stats import Confusion


tfidf_model =TfidfVectorizer(tokenizer=casual_tokenize)

pd.options.display.width = 120 #display wide column of SMS text within a pandas DataFrame printout

sms = get_data('sms-spam')
index = ['sms{}{}'.format(i,'!'*j) for (i,j) in zip(range(len(sms)),sms.spam)]

print(type((sms)))
print(sms.head(5))
# original line
###### page 108

#modified line
#print(sms.columns)
for col in sms.columns:
	print(col)
#this line is redundant and therefore no needed. 
#sms = pd.DataFrame(sms.text, columns=sms.columns, index=index)
# there are some NaN values in the span column. They need to be converted to 0 before sms['spam'] = sms.spam.astype(int) 
#sms['spam'] = sms['spam'].fillna(0)
#print(sms.head(5))
sms['spam'] = sms.spam.astype(int)

print(len(sms))
print('total of spam ',sms.spam.sum())
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()
print(tfidf_docs.shape)
# first we need to compute the centroid of the two classes
# this is use to select only the spam rows from a numpy.array or pandas.DataFrame
mask = sms.spam.astype(bool).values
spam_centroid = tfidf_docs[mask].mean(axis=0)
ham_centroid = tfidf_docs[~mask].mean(axis=0)

print(spam_centroid.round(2))
print(ham_centroid.round(2))
spamminess_score =tfidf_docs.dot(spam_centroid-ham_centroid)
print(spamminess_score.round(2))

sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1,1))
sms['lda_predict'] = (sms.lda_score>0.5).astype(int)
print(sms.head(6))
print('{} well classified instances'.format((1.0 - (sms.spam - sms.lda_predict).abs().sum()/len(sms)).round(3)))
cfn_mat = Confusion(sms['spam lda_predict'.split()])
print(cfn_mat)
