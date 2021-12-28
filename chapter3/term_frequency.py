from nltk.tokenize import TreebankWordTokenizer
from collections import Counter, OrderedDict
from nlpia.data.loaders import kite_text
from nltk.corpus import stopwords
from nlpia.data.loaders import harry_docs as docs 
import copy

stop_words =stopwords.words('english')
sentence = """The faster Harry got to the  store, the faster Harry, the faster, would get home."""

tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
print(tokens)
bag_of_words = Counter(tokens)
print(bag_of_words)
print('Most common words')
print(bag_of_words.most_common(4))

## calculating the term frequency of harry 
times_harry_appears = bag_of_words['harry']
num_unique_words = len(bag_of_words)
tf = times_harry_appears/num_unique_words
print('Term frequency {}'.format(round(tf,4)))

###

tokens = tokenizer.tokenize(kite_text.lower())
tokens = [x for x in tokens if x not in stop_words]
kite_counts = Counter(tokens)
print(kite_counts)

### vectorising the text

document_vector = []
doc_length = len(tokens)

for key, value in kite_counts.most_common():
	document_vector.append(value/doc_length)

print(document_vector)

## page 77
doc_tokens = []
for doc in docs:
	doc_tokens +=[sorted(tokenizer.tokenize(doc.lower()))]
print(len(doc_tokens[0]))
all_doc_tokens = sum(doc_tokens,[])
print(len(all_doc_tokens))
lexicon = sorted(set(all_doc_tokens))
print(len(lexicon))
print(lexicon)

zero_vector = OrderedDict((token,0) for token in lexicon)
print(zero_vector)

doc_vectors = []
for doc in docs:
	vec = copy.copy(zero_vector)
	tokens = tokenizer.tokenize(doc.lower())
	token_counts = Counter(tokens)
	for key, value in token_counts.items():
		vec[key] = value/len(lexicon)
	doc_vectors.append(vec)
