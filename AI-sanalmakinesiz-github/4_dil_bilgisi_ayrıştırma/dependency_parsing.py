"""

Amac:
    -bir türkce cümledeki kelimeler arasında ki bağımlılık (dependency) ilişkilerini bulmak
    - cümlede hangi kelime diğerine bağlı,kök kelime hangisi ,özne (subject), nesne (object) gibi ilişkileri görmek 
    - bunun icin stanza kütüphanesini kullanalım
kurulum : 
pip install stanza
    
    """
import stanza
nlp = stanza.Pipeline(lang = "tr",processors = "tokenize,pos,lemma,depparse")

sentence = "Aytu sabaha okula hızlıca yürüdü."

doc = nlp(sentence)

print(f"{"Kelime":12}{"Lemma":12}{"POS":10}{"head":12}{"bağlantı (relation)"}")

for sent in doc.sentences:
    for word in sent.words:
        #word.head kelimenin bağlı olduğu kelimenin id sini veriyor
        #head = 0 ise bu kelimenin köküdür
        head_text="ROOT" if word.head == 0 else sent.words[word.head - 1].text


        print(f"{word.text:12}{word.lemma:12}{word.upos:10}{head_text:12}{word.deprel}")

"""
Kelime      Lemma       POS       head        bağlantı (relation)
Aytu        Aytu        PROPN     yürüdü      nsubj
sabaha      sabah       NOUN      yürüdü      obl
okula       okul        NOUN      yürüdü      obl
hızlıca     hızlıca     ADV       yürüdü      advmod
yürüdü      yürü        VERB      ROOT        root
.           .           PUNCT     yürüdü      punct
"""