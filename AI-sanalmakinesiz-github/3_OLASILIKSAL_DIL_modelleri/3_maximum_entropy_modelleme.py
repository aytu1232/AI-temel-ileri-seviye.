"""
amaç:
maximum entropy ile duygu analizi (sınıflandırıcı)

adımlar:
    - veri seti
    - nltk max ent classifier model
    - yeni cümlelerden öznitelik çıkarımı ve sınıflandırma
    - ayırt edici özellikleri incele

pip install nltk
"""
from nltk.classify import MaxentClassifier

train_data = [
    ({"love": True, "amazing": True, "great": True, "terrible": False, "bad": False}, "positive"),
    ({"hate": True, "terrible": True, "awful": True, "love": False}, "negative"),
    ({"happy": True, "joy": True, "good": True, "sad": False}, "positive"),
    ({"sad": True, "depressed": True, "bad": True, "happy": False, "amazing": False}, "negative"),
    ({"wonderful": True, "pleasant": True, "nice": True, "awful": False}, "positive"),
    ({"angry": True, "hate": True, "upset": True, "good": False, "great": False}, "negative")
]

classifier = MaxentClassifier.train(train_data, max_iter = 15)

classifier.show_most_informative_features(5) #ayırt edici özellikleri gösterir

#öznitelik cıkarımı icin yardımcı
def extract_features(sentence: str, vocab= None):
    tokens = sentence.lower().split() # basit tokenization
    if vocab is None:
        vocab = ["love", "amazing", "great", "good", "happy", "joy", "wonderful", "pleasant", "nice",
                 "hate", "terrible", "awful", "bad", "sad", "depressed", "angry", "upset"]
    return {word: (word in tokens) for word in vocab}

test_sentences = [
    "I love this product it is amazing and wonderful",
    "This is bad I hate the design it is awful",
    "The movie was good and pleasant overall",
]

for i, sent in enumerate(test_sentences, 1):
    feats = extract_features(sent)
    label = classifier.classify(feats)
    prob_dist = classifier.prob_classify(feats)  # sınıfın olasılık dağılımı
    p_pos = prob_dist.prob("positive")
    p_neg = prob_dist.prob("negative")
    print(f"Test {i} {sent}")
    print(f" Predicted: {label}, P(pos)={p_pos}, P(neg) = {p_neg}")