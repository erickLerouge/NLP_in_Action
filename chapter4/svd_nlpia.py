from nlpia.book.examples.ch04_catdog_lsa_sorted import lsa_models, prettify_tdm
import numpy as np
import pandas as pd

bow_svd, tfidf_svd = lsa_models()
print(prettify_tdm(**bow_svd))
tdm = bow_svd['tdm']
print(tdm)
U,s,Vt = np.linalg.svd(tdm)
df_U = pd.DataFrame(U,index=tdm.index).round(2)
print(df_U.head())	

print(s.round(2))
df_Vt = pd.DataFrame(Vt).round(2)
print(df_Vt.head())
