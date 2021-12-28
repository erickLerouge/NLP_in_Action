from nlpia.data.loaders import get_data
import pandas as pd
from nltk.tokenize import casual_tokenize# handles slangs, unusual ponctuation and emoticons
from collections import Counter
from sklearn.naive_bayes import MultinomialNB


movies = get_data('hutto_movies')
print(movies.head().round(2))

#so now we tokenize the text and put it into a dataframe

pd.set_option('display.width',75) # makes dataframes visualisation nicer in the consoole
bags_of_words = []

for text in movies.text:
	bags_of_words.append(Counter(casual_tokenize(text)))

df_bows = pd.DataFrame.from_records(bags_of_words)
df_bows = df_bows.fillna(0).astype(int)

print('The size of the df is {} x {}'.format(df_bows.shape[0], df_bows.shape[1]))

#lets start the ML thing

nb = MultinomialNB()
nb = nb.fit(df_bows,movies.sentiment>0)
# this line is as it appears in the book, however the output of predict_proba represnets an array/matrix with the number of classes we trained the model
# doing this
#movies['predicted_sentiment'] = nb.predict_proba(df_bows)* 8 - 4
# will through an error since it tries to put a two-columns array into one column in one colum of the dataframe
#mySolution
tmp = nb.predict_proba(df_bows)* 8 - 4
movies['predicted_sentiment'] = tmp[:,1]
#continue with the original code
movies['error'] = (movies.predicted_sentiment - movies.sentiment).abs()
movies.error.mean().round(1)
movies['sentiment_ispositive'] = (movies.sentiment>0).astype(int)
movies['predicted_ispositiv'] = (movies.predicted_sentiment>0).astype(int)

print(movies.head(8))

print('Rating correct {} of the time'.format((movies.predicted_ispositiv==movies.sentiment_ispositive).sum()/len(movies))) 
