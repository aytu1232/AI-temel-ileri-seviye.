import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re #regular expression >(veri temizleme,arama)
import nltk
from nltk.corpus import stopwords
from collections import Counter
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
veri = pd.read_csv("IMDB Dataset.csv")

print(veri.head())

yorumlar = veri["review"]

etiketler = veri["sentiment"]

#metin temizleme
def metin_temizleme(metin):
    # metin = metin.lower()
    metin = re.sub(r"\d+","",metin)

    metin = re.sub(r"[^\w\s]","", metin)

    # metin = "".join([kelime for kelime in metin.split() if len(kelime) > 2])#stop wordsüz 

    metin = " ".join([kelime for kelime in metin.lower().split() if len(kelime) > 2 and kelime not in stop_words])
    return metin

temizlenmis_yorumlar = [metin_temizleme(y) for y in yorumlar]

veri["temizlenmis_yorumlar"] = temizlenmis_yorumlar
bow_modeli = CountVectorizer()
print(veri.head())

yorum_vektorleri = bow_modeli.fit_transform(temizlenmis_yorumlar[:100])
kellime_sayilari = yorum_vektorleri.sum(axis=0).A1
kelime_kumesi = bow_modeli.get_feature_names_out()
vektor_temsili = yorum_vektorleri.toarray()
print(f"Vektör temsili: {vektor_temsili}")

df_bow= pd.DataFrame(vektor_temsili , columns = kelime_kumesi)
print(df_bow.head())

kelime_frekansi = dict(zip(kelime_kumesi,kellime_sayilari))
en_cok_gecen_6 = Counter(kelime_frekansi).most_common(5)

print(f"en cok gecen 5 kelime:{en_cok_gecen_6}")