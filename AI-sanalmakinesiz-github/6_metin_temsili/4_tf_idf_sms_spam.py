"""
Amac : sps spam veri seti üzerinden tfidf ile analiz

    Adımlar:
        1.csv dosyasından sms verisini yükleyip
        2.tf idf vektöreizer ile sms verisini sayısal vektörlere dönüştürücez
        5.oralamaa tf idf skorunu hesaplıcaz
        5.sonucları df 2 aktarıp en yüksek skora sahip kelimeleri bulcaz.
        """

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

#csv dosyasını yükledik
veri = pd.read_csv("sms_spam.csv")

#sms mesajlarını alıyoruz
mesajlar = veri["text"]

#tf idf
tf_idf_model = TfidfVectorizer()

#mesaj dönüşümü
mesaj_vektörleri = tf_idf_model.fit_transform(mesajlar)

#stop words cıkarma 
# nltk.download('stopwords')
# stop_words_nltk = stopwords.words("english")
# tf_idf_model = TfidfVectorizer(stop_words=stop_words_nltk)
# mesaj_vektörleri = tf_idf_model.fit_transform(mesajlar)
"""
     kelime  ortalama_tf_idf
1804   call         0.019740
5461     ok         0.017985
3483    get         0.013877
2138   come         0.011487
3644     gt         0.011329
"""
#stop words cıkarma2
tf_idf_model = TfidfVectorizer(stop_words="english")
mesaj_vektörleri = tf_idf_model.fit_transform(mesajlar)
"""
     kelime  ortalama_tf_idf
5377     ok         0.018593
4566     ll         0.014328
4250   just         0.012961
2108   come         0.012079
3584     gt         0.011628
"""

#kelime kümemizi oluşturuyoruz
kelime_kümesi = tf_idf_model.get_feature_names_out()

#ortalama skor hesaplama
tf_idf_skorlari = mesaj_vektörleri.mean(axis = 0).A1

#sonucları df icerisinde yaz
df_tf_idf = pd.DataFrame({"kelime": kelime_kümesi , "ortalama_tf_idf":tf_idf_skorlari})
df_tf_idf_sirali = df_tf_idf.sort_values(by = "ortalama_tf_idf", ascending = False)

print (df_tf_idf_sirali.head())

"""     
kelime  ortalama_tf_idf
8668    you         0.044201
7806     to         0.037120
7674    the         0.026506
4114     in         0.022001
4968     me         0.021279
"""