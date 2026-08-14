"""
Amac:
    -1bir cümledeki her kelimenin dilbilgisel türünü bulmak
    -spacy kütüphanesi ile her token icin pos etiketi bulalım
    
adımlar:
    -spacy modelini yüklicez
    -örnek cümle oluşturucaz ve nlp modelinden geciricez
    -her kelimenin pos etiketini yazdır.

    pip install spacy
    python -m spacy download en_core_web_sm
    """

import spacy

nlp_model = spacy.load("en_core_web_sm")

sentence = "Can you recommend a good restaurant in London."

doc = nlp_model(sentence)

for token in doc:
    print(f"{token.text:12}{token.pos_}")

"""
Can         AUX (yardımcı fiil)
you         PRON (zamir)
recommend   VERB (fiil)
a           DET(belirtec)
good        ADJ(sıfat)
restaurant  NOUN(isim)
in          ADP(edat)
London      PROPN(özel ad)
.           PUNCT (noktalama işareti)
"""
import stanza

stanza.download("tr")

nlp = stanza.Pipeline("tr")

sentence =["Bunu yazan tosun okuyana kosun.","Okuyan çok koymuş tosun hoplaya hoplaya top olmuş."]


for tekcümle in sentence:
    doc = nlp(tekcümle)
    for sent in doc.sentences:
        for word in sent.words:
            print(f"{word.text:15}{word.upos:12}")