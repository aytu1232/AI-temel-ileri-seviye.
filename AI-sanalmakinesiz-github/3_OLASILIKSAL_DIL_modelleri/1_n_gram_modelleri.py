"""
n-gram uygulaması
    1-kücük veri seti oluşturma
    2-veri temizleme (kücük harfe cevirme) ve tokenizasyon
    3-bigram ve trigram oluştur(nltk)
    4-frekans sayımı
    5-koşullu olasılık hesaplama
    örn:"i love..." sonrasında you ya da apple gelme olasılıkları hesaplama

pip install nltk
    """

import nltk
from nltk.util import ngrams #n-gram icin gerekli fonksiyon
from nltk.tokenize import word_tokenize #tokenizasyon icin
from collections import Counter #frekans sayacı

#ilk calışma ici tokenizasyon modellerini indirme
# nltk.download("punkt")
# nltk.download("punkt_tab")

#corpus oluşturma (kücük veri seti)
raw_corpus = [
    "I love apples",
    "I love you",
    "We love NLP",
    "You love me",
    "He lover apples",
    "They love apples",
    "I love coding and you love learning",
    "We love machine learning"
    "You love apples and bananas",
    "I truly love natural language processing"
]

#veri temizleme ve tokenizasyon
tokenized_sents=[word_tokenize(sent.lower())for sent in raw_corpus]

#n gram üretimi
bigram_list=[]
for toks in tokenized_sents:
    bigram_list.extend(list(ngrams(toks, 2))) #bi gram 
print(f"bigram_list: \n{bigram_list}")
trigram_list=[]
for toks in tokenized_sents:
    trigram_list.extend(list(ngrams(toks, 3)))#trigram
print(f"bigram_list: \n{trigram_list}")

#frekans sayımları
bigram_counts = Counter(bigram_list) #count(w1,w2)
trigram_counts = Counter(trigram_list)#count(w1,w2,w3)

print(f"en sık 5 bigrams:{bigram_counts.most_common(5)}")
print(f"en sık 5 trigrams:{trigram_counts.most_common(5)}")

#koşullu olasılık hesaplama
#amac:"i love" bigramından sonra gelecek kelimenin olasılığını hesaplamak
#P(kelime | "i","love") = count("i","love", kelime)/ count("i","love")

context_bigram = ("i","love")

#deneyeceğimiz aday kelimeler

candidates = ["you","apples","nlp","coding"]

#olasılık hesabı
def conditional_prob(w1,w2,w3):
    numerator = trigram_counts.get((w1,w2,w3),0) #count(w1,w2,w3) -> count ("i","love",kelime)
    denominator = bigram_counts.get((w1,w2),0) #count (w1,w2) -> count("i","love")
    if denominator == 0:
        return 0
    return numerator/ denominator

print(f"bağlam:{context_bigram}")
for cand in candidates:
    p = conditional_prob(context_bigram[0], context_bigram[1], cand)
    print(f"P({cand!r} | {context_bigram[0]!r}, {context_bigram[1]!r}) = {p:.4f}")
