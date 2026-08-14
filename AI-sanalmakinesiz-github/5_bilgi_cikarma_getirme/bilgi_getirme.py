"""
Amaç:
    - Bu uygulama, BERT modelini kullanarak metin benzerliği (semantic similarity) analizi gerçekleştirecek.
    - Bir sorgu cümlesi (query) ile bir dizi belgenin (documents) anlama ne kadar benzer olduğu ölçülür.
    - Her metin BERT modelinden elde edilen embedding (vektör temsili) ile temsil edilir.
    - Benzerlik ölçümü için cosine similarity kullanılır.

İzlenecek adımlar:
    1. gerekli kütüphaneleri içeriye aktar.
    2. Bert modelini ve tokenizeri yükle
    3. Örnek belge (documents) ve sorgu cümlesi oluştur
    4. Her metni vektör haline getir yani embedding yap
    5. Sorgu ve belgeler arasında ki benzerliği hesapla
    6. en benzer belgeyi belirle ve yazdır

Kurulumlar:
    pip install transformers torch scikit-learn numpy
"""

from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity #benzerlik hesaplama

#modeo ve tokenizer yükleme 
model_name = "bert-base-uncased" #kücük boyutlu ingilizce metinlerde kullanılan model
tokenizer = BertTokenizer.from_pretrained(model_name) #tokenizer bert base uncades ile oluşturulur
model = BertModel.from_pretrained(model_name)#önceen eğitilmiş model yüklenir.

#veri oluşturma belgeleri oluşturma
documents = [
    "Machine learning is a field of artificial intelligence",
    "Natural language processing involves understanding human language",
    "Artificial intelligence encompasses machine learning and natural language processing",
    "Deep learning is a subset of machine learning",
    "Data science combines statistics, data analysis and machine learning",
    "I like shopping"
]

#kullanıcı sorgusu
query ="Get information about natural language processing"

def get_embedding(text):
    """
    verilen bir metni BERT ile sayısal vektör yapar.
    tokenization yapılır
    model calışır
    embedding yapılır
    """
    #pytorch formatinda tensorlere cevirme
    #truncation = 512 toekn sınırını aşarsa metni keser
    #padding = giriş uzunluğu eşitleme
    inputs = tokenizer(text,return_tensors ="pt",truncation = True , padding = True)

    #modeli calıştırma
    outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    embedding = last_hidden_state.mean(dim=1)

    return embedding.detach().numpy()

doc_embeddings = np.vstack([get_embedding(doc) for doc in documents]) #her belge icin embedding
query_embedding = get_embedding(query)

#benzerlik hesaplama 1 ise benzer 0 ise alakasız

similarities =cosine_similarity(query_embedding, doc_embeddings)

#sonuc yazdır
for i, score in enumerate(similarities[0]):
    print(f"Document: {documents[i]}\nSimilarity Score: {score:.4f}\n")

#en yüksek benzerliğe sahip belge bulma
most_similar_index = similarities.argmax()
print(f"Most similar document: {documents[most_similar_index]}")