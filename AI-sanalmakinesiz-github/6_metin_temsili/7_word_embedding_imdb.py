import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

import nltk
from nltk.corpus import stopwords



veri = pd.read_csv("IMDB Dataset.csv")
yorumlar = veri ["review"]
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def metin_temizle(metin):

    metin = metin.lower()

    metin = re.sub(r"\d+","",metin)

    metin = re.sub(r"[^\w\s]","",metin)

    kelimeler = metin.split()
    kelimeler = [kelime for kelime in kelimeler if kelime not in stop_words]

    kelimeler = [kelime for kelime in kelimeler if len(kelime) > 2]

    temiz_metin =" ".join(kelimeler)
    return temiz_metin

temizlenmis_yorumlar = [metin_temizle(y) for y in yorumlar]

#tokenizasyon
tokenize_yorumlar = [simple_preprocess(y) for y in temizlenmis_yorumlar]

#eğitim parametleri
word2vec_model = Word2Vec(
    sentences = tokenize_yorumlar, vector_size = 50 , window = 5 , min_count=1 ,sg=0
)

kelime_vektorleri = word2vec_model.wv

kelimeler = list(kelime_vektorleri.index_to_key)[:500]

vektorler = [kelime_vektorleri[w] for w in kelimeler]

#eğitim
kmeans = KMeans(n_clusters=2)
kmeans.fit(vektorler)
kume_etiketleri = kmeans.labels_

pca = PCA(n_components=2)
indirgenmis_vektorler = pca.fit_transform(vektorler)

plt.figure()
plt.scatter(indirgenmis_vektorler[:,0],indirgenmis_vektorler[:,1],c = kume_etiketleri,cmap ="viridis")

merkezler = pca.transform(kmeans.cluster_centers_)
plt.scatter(merkezler[:,0],merkezler[:,1],c = "red",marker = "x" , s = 170 , label = "küme merkezi")
for i, kelime in enumerate(kelimeler):
    plt.text(indirgenmis_vektorler[i, 0], indirgenmis_vektorler[i, 1], kelime, fontsize=10)
plt.title("word2vec + PCA ile 2 boyuta indirgeme + kmeans ile kümele")
plt.legend()
plt.show()