"""
Adımlar :
    1. kücük belge oluşturucaz
    2. TF - IDF vektörizer ile belgeleri sayısal vektöre dönüştürücez
    3.Kelime kümesi cıkartıcaz
    4.Belgelerin tf idf vektör temsillerini elde edicez.
    5.Tüm belgeler icin kelimelerin ortalama tf idf değerlerini hesaplıcaz.
    """

import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

belgeler = [
    "köpek cok tatlı bir hayvandır.",
    "Köpek ve kuşlar cok tatlı hayvanlardır.",
    "İnekler süt üretirler."
]

tfidf_modeli = TfidfVectorizer()

belge_vektörleri = tfidf_modeli.fit_transform(belgeler)


kelime_kumesi = tfidf_modeli.get_feature_names_out()

vektör_temsili = belge_vektörleri.toarray()

print(f"Tf IDF matrisi : {vektör_temsili}")

"""
Tf IDF matrisi : 
 [[0.51741994 0.3935112  0.51741994 0.         0.         0.3935112
  0.         0.         0.3935112  0.         0.        ]
 [0.         0.34949812 0.         0.45954803 0.45954803 0.34949812
  0.         0.         0.34949812 0.45954803 0.        ]
 [0.         0.         0.         0.         0.         0.
  0.57735027 0.57735027 0.         0.         0.57735027]]
"""

#bu formatı okunabilir hale getiricez

df_tfidf = pd.DataFrame(vektör_temsili, columns = kelime_kumesi)
print(df_tfidf)
"""
       bir       cok  hayvandır  hayvanlardır    kuşlar     köpek   nekler      süt     tatlı        ve  üretirler
0  0.51742  0.393511    0.51742      0.000000  0.000000  0.393511  0.00000  0.00000  0.393511  0.000000    0.00000
1  0.00000  0.349498    0.00000      0.459548  0.459548  0.349498  0.00000  0.00000  0.349498  0.459548    0.00000
2  0.00000  0.000000    0.00000      0.000000  0.000000  0.000000  0.57735  0.57735  0.000000  0.000000    0.57735
"""

#her kelimenin belgeler arasındaki ortalama tf idf değerlerini hesapla

ortalama_tf_idf = df_tfidf.mean(axis = 0)
print(f"Kelimelerin ortalama tf idf değerleri : {ortalama_tf_idf}")

"""
Kelimelerin ortalama tf idf değerleri : bir             0.172473
cok             0.247670
hayvandır       0.172473
hayvanlardır    0.153183
kuşlar          0.153183
köpek           0.247670
nekler          0.192450
süt             0.192450
tatlı           0.247670
ve              0.153183
üretirler       0.192450
dtype: float64
"""