"""
bag of words B.O.W
Amaç:
    - Metin temsili: Bunu yaparken bag of words kullanalım: metin listesinden -> sayısal vektörlere
    - Bunun için sklearn kütüphanesinden CountVectorizer: kelimelerin kaç defa geçtiğini sayar ve vektöre temsiline dönüştürür

Sonuç:
    - kelime kümesi (vocabulary)
    - her metin listesi sayısal vektörler ile temsil edilecek

pip install scikit-learn
"""

#import libraries
from sklearn.feature_extraction.text import CountVectorizer

#veri seti oluşturuyoruz
#kücük veri seti
dokumanlar = [
    "kedi bahcede",
    "kedi evde"
]
#bow - bag of words
kelime_sayac = CountVectorizer()

#dokümanları sayısal vektörlere cevirme
dokuman_vektorleri = kelime_sayac.fit_transform(dokumanlar)

#kelime listesi (bulunan)
kelime_kumesi = kelime_sayac.get_feature_names_out()
print(f"kelime kümesi: {kelime_kumesi}")

#vektor temsili
vektor_temsili = dokuman_vektorleri.toarray()
print(f"Vektör temsili: {vektor_temsili}")

#sonucları değerlendirme

"""
kelime kümesi: ['bahcede' 'evde' 'kedi']
Vektör temsili: [[1 0 1]
 [0 1 1]]
 """