"""
doğal dil işleme icin temel adık olan tokenizasyon
-kelime tokenizasyonu
-cümle tokenizasyonu

pip install nltk (natural language tool kit)
"""

import nltk

# nltk.download("punkt") #kelime ve cümle tokenizasyonu icin
# nltk.download("punkt_tab") #nltk yeni versiyonlarında ek olarak indirilmesi gerekiyor.

#örnk
raw_text = "Merhaba dünya ! Bu bir NLP eğitim örneğidir. Sen naıslsın?, Hella ,hi ..."

#kelime tokenizasyonu 
word_tokens = nltk.word_tokenize(raw_text)
print(f"Word tokens : {word_tokens}")

#cümle tokenizasyon
sentence_tokens = nltk.sent_tokenize(raw_text)
print(f"sentence tokenleri : {sentence_tokens}")