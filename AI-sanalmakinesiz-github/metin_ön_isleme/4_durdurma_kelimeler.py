"""
önemsiz kelimeleri cıkarma yöntemleri
-ingilizce stop word kullanma
-türkce stop words
-kütüphane kullanmadan manuel olarak stop words cıkarma
pip install nltk (venvde kurulu hata verirse tekrar kur.)
"""

import nltk
from nltk.corpus import stopwords

#ilk calıştırma da stop words veri seti indirme
nltk.download("stopwords")

#ingilizce stop wordsler

stop_words_eng = set(stopwords.words("english"))

#örnk metin
eng_text = "This is just a simple example to show how stop words can be removed from the sentences basically expamle"
eng_text_list = eng_text.split()
print(eng_text_list)

#eğer stop word de yoksa and,or,if vb yeni listeye ekleme
filtered_words_eng = [word for word in eng_text_list if word.lower() not in stop_words_eng]
print(f"orjinal:{eng_text}")
print(f"filtrelenmiş : {filtered_words_eng}")

stop_words_tr = set(stopwords.words("turkish"))

tr_text = "Merhaba bugün NLP dersi öğreniyorum kendimi geliştirmek ve yapay zeka mühendisi olmak istiyorum."
tr_text_list = tr_text.split()

#stop word temizleme
filtered_words_tr = [word for word in tr_text_list if word.lower() not in stop_words_tr]
print(f"orjinal:{tr_text}")
print(f"filtrelenmiş : {filtered_words_tr}")

#kendi stop  word oluşturma
custom_tr_stopwords =["icin","bu","ile","mı","cok","ve"]
custom_text= "bu bir denemedir ve amacımız bu metindeki bazı kelimeleri cıkarmak mı acaba"
custom_text_list=custom_text.split()
filtered_custom_words_tr =[word for word in custom_text_list if word.lower() not in custom_tr_stopwords]

print(f"orjinal {custom_text}")
print(f"custom filtre {filtered_custom_words_tr}")