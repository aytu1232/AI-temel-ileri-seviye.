"""
kök veya temel form (lemma) bulma
-stemming: porter stemmer
-lemmatization: word net lemmatizer (gövde bulma)

pip install nltk
"""

import nltk
nltk.download("wordnet") #lemmatization icin gerekli (gövdeleme)
nltk.download("omw-1.4")#wordnet icin ek dil desteği

#stemming
from nltk.stem import PorterStemmer #ingilizce icin stemmer algoritması

stemmer = PorterStemmer() #porter stemmer nesnesini oluşturur

word_stem = ["playing","played","plays","happily","happier","studies","studying"]
stems = [stemmer.stem(w) for w in word_stem] #kökler
print(f"orjinal : {word_stem}")
print(f"kökler:{stems}")


#lemmatization 
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()#lematizer nesnesini oluşturma

words_lemma =["running","ran","gone","Better","children"]
lemmas = [lemmatizer.lemmatize(w) for w in words_lemma]
print(f"orjinal:{words_lemma}")
print(f"lemmas: {lemmas}")