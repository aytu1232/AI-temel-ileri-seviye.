"""
Partof Speech(POS) sözcük türü etiketleme işlemi.
Hidden markov model tabanlı bir etiketleyici:
    1-Kücük el ile tanımlanmış bir eğitim veri kümesini HMM(hiddeb markov model) eğitimi ve testleri yapalım
    2-NLTK connl12000 veri seti ile kapsamlı HMMM eğiti

aytuğ okula gitti . Aytuğ -> özne, okula -> nesne, gitti -> yüklem.
pip install nltk numpy
"""

import nltk
from nltk.tag import hmm # gmm tabanlı pos etiketleyici mödül

import numpy as np

#el ile tanımlı veri seti
"""
mrnek eğitim veri seti: (kelime -etiket) ciftlerinden oluşan cümle listesi
PRP: zamir
VBP: geniş zaman fiil
DT: belirtec
MN: isim
VBZ: 3.tekil şahıs fiil
"""

toy_train_data = [
    [("i", "PRP"), ("am", "VBP"), ("a", "DT"), ("developer", "NN")],
    [("you", "PRP"), ("are", "VBP"), ("a", "DT"), ("student", "NN")],
    [("he", "PRP"), ("is", "VBP"), ("an", "DT"), ("engineer", "NN")]
]

#HMM oluşturma (Hidden Markov Model)

toy_trainer = hmm.HiddenMarkovModelTrainer()

#training 
toy_hmm_tagger = toy_trainer.train(toy_train_data) #train: verilen etkiketli veriden HMM etiketlerini öğrenir

toy_test_sentence_1 = "I am an engineer".lower().split()#tokenlere ayırır
toy_tags_1=toy_hmm_tagger.tag(toy_test_sentence_1)#tag: her kelime icin en olası POS etiketini döndürür
print(f"Test cümlesi 1: {toy_test_sentence_1}")
print(f"etiketler : {toy_tags_1}")

toy_test_sentence_2 = "He is a developer".lower().split()#tokenlere ayırır
toy_tags_2=toy_hmm_tagger.tag(toy_test_sentence_2)#tag: her kelime icin en olası POS etiketini döndürür
print(f"Test cümlesi 2: {toy_test_sentence_2}")
print(f"etiketler : {toy_tags_2}")


from nltk.corpus import conll2000

nltk.download("conll2000")

big_train_data = conll2000.tagged_sents("train.txt") #eğitim veri seti
big_test_data= conll2000.tagged_sents("test.txt")#test verisi



#hmm tanımlama ve eğitme 
big_trainer = hmm.HiddenMarkovModelTrainer()
big_hmm_tagger = big_trainer.train(big_train_data)

#test
big_test_sentence_1 = " We enjoy learning machine learning concepts".split()
big_tags_1 = big_hmm_tagger.tag(big_test_sentence_1)
print(f"test cümlesi 1: {big_test_sentence_1}")
print(f"etiketler {big_tags_1}")