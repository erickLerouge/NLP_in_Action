#this code only helps to understand how to create a tf-idf matrix representation 
from nlpia.data.loaders import kite_text, kite_history
from nltk.tokenize import TreebankWordTokenizer
from nlpia.data.loaders import harry_docs as docs
from collections import Counter
tokenizer = TreebankWordTokenizer()
from sklearn.feature_extraction.text import TfidfVectorizer


kite_intro = kite_text.lower()
intro_tokens = tokenizer.tokenize(kite_history)
kite_history = kite_history.lower()

history_tokens = tokenizer.tokenize(kite_history)
intro_total = len(intro_tokens)
print(intro_total)

history_total = len(history_tokens)

print(history_total)

intro_tf = {}
history_tf ={}
intro_counts = Counter(intro_tokens)
intro_tf['kite'] = intro_counts['kite']/intro_total

## feature extraction with tfidf

corpus = docs
vectorizer = TfidfVectorizer(min_df=1)
model = vectorizer.fit_transform(corpus)
print(model.todense().round(2))
