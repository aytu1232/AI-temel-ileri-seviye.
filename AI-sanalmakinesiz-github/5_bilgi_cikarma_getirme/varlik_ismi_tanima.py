"""
pip install pandas spacy
python -m spacy download en_core_web_sm
"""

import pandas as pd
import spacy

# psacy ingilizce modeli 

# nlp_model = spacy.load("en_core_web_sm")
nlp_model = spacy.load("tr_core_news_md")

sample_text = "Atatürk, 1923 yılında Ankara'da Türkiye Cumhuriyeti'ni kurdu. Şu an Microsoft ve Google gibi teknoloji devleri İstanbul'da yeni ofisler açıyor"
# sample_text = "Alice works at Amazon and lives in London. She visited the British Museum last weekend."
#spacy e metni veriyoruz tokenizasyon pos tagging ve ner yapar
doc = nlp_model(sample_text)

for entity in doc.ents:
    print(entity.text, entity.label_)

#varlıkların lemma biligi ile birlikte saklama

entities_list = [(entity.text, entity.label_, entity.lemma_ )for entity in doc.ents]

df_entities = pd.DataFrame(entities_list, columns=["text","type","lemma"])

print(df_entities)

"""
Alice PERSON
Amazon ORG
London GPE
the British Museum ORG
last weekend DATE
"""