from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA
import seaborn as sns

#from matplotlib import pyplot as plt
#from nlpia.plots import offline_ploty_scatter_bubble

pca = PCA(n_components = 2)
word_vectors = KeyedVectors.load_word2vec_format('/Users/velazquezerick/Documents/NLP_in_action/GoogleNews-vectors-negative300.bin',binary=True)
#print(len(word_vectors.wv))
### this visualization is made on data that correspond to the information of cities around the world
### it was impossible to load the information for the cities. So I will use the embeddings I create from the movie mobidick
from nlpia.data.loaders import get_data
#cities = get_data('cities')
#print(cities.head(1).T)
some_words = []
tmp = ['man','whale','demon']
for word in tmp:
	for t in word_vectors.most_similar(word):
		some_words.append(word_vectors[t[0]])
#us_300D = get_data('cities_us_wordvectors')
data_2d = pca.fit_transform(some_words)

sns.set_theme(style='whitegrid')
cmap = sns.cubehelix_palette(rot = -.2, as_cmap=True)

g = sns.replot(data = data_2d, x="distance",)
