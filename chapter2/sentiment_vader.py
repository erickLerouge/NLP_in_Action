from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sa = SentimentIntensityAnalyzer()
#print(sa.lexicon)
print(sa.polarity_scores('Python is very readable and it is great for NLP'))
print(sa.polarity_scores('Python is not a bad choice for most applications'))

# trying for a couple of sentences:
corpus = ['Absolutely perfect! Love it! :-) :-) :-)', "Horrible! completely useless. :(", 'It was ok. Some good and somebad things']
for doc in corpus:
	scores = sa.polarity_scores(doc)
	print('{:+}: {}'.format(scores['compound'],doc))
