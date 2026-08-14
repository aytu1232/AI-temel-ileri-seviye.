"""
Amac:
    Temel veri temizleme adımları:
        -fazla boşlukların kaldırılması
        -büyük harflerin kücük harfe dönüştürülmesi
        -noktalama işaretlerinin kaldırılması
        -özel karakterlerin kaldırılması
        -yazım hataları düzeltilmesi
        -html etiketlerinden sadece düz metin elde etme

pip install textblob beautifulsoup4
"""
#fazla boşlukları temizle

raw_text = "Python,      NLP!       2044"# fazla boşluk iceren metin örneği
#büyük kücük harf dönüşümü
normalized_text_1 = raw_text.split()
print(raw_text.split())
normalized_text_1 = " ".join(raw_text.split())
print(f"fazla boşluklar kaldırılmış hali \n{normalized_text_1}")
#büyük kücük harf dönüşümü
import string
raw_text="HeLLo OpenAI 2026"
normalized_text_2=raw_text.lower()
print(f"kücük harf:{normalized_text_2}")
#noktalama işaretlerinden kurtulma 
raw_text = "AI , Machine-Learning! 2037?"
normalized_text_3 = raw_text.translate(str.maketrans("", "", string.punctuation))
print(f"noktalama işaretlerinden kurtul : {normalized_text_3}")

#öz<el karakterlerden kurtul (%,@,#,/ vb.)
import re
raw_text = "Deep@Learning% is*great 2038."
normalized_text_4 =re.sub(r"[^A-Za-z0-9\s]","",raw_text)
print(f"özel karakterlerden kurtul:{normalized_text_4}")

#yazım hatalarını düzeltme
from textblob import TextBlob

raw_text="It is amazng in 2039"
normalized_text_5 = TextBlob(raw_text).correct()#yazım hatalarını düzeltir
print(f"yazım hatası düzeltilmiş:{normalized_text_5}")

#html etiketlerından düz metin
from bs4 import BeautifulSoup
raw_html = "<div> 2040 hello</div>"
normalized_text_6 = BeautifulSoup(raw_html,"html.parser").get_text()
print(f"HTML:{normalized_text_6}")