"""
Amaç:
    - Bu uygulamanın amacı, Hugging Face'in Transformers kütüphanesini kullanarak bir metin özetleme (bilgi çıkarma) pipeline'ı oluşturmak.
    - Verilen bir metni kısa bir özet haline getirmek yani metinden bilgi çıkarmak
    - Burada modelimizi önceden eğitilmiş büyük dil modeli olarak kullanalım, metnin ana fikrini koruyarak kısa bir özet üretir.

İzlenecek adımlar:
    - gerekli kütüphaneleri içeri aktar
    - Özetleme (summarization) pipeline'ı oluştur
    - uzun bir metin tanımla (gpt ile hikaye oluştururuz)
    - modeli çalıştırarak özet oluştur
    - sonucu ekrana yazdırma

kurulumlar:
    pip install transformers torch
    pip install accelerate
"""

from transformers import pipeline 

# özetleme pipeline yükle

summarizer = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device_map="auto") # summarization modeli özetleme yapsın ,llm modeli cağırısn
# summarizer = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct") #pip install accelerate yapmadan calıştırmak icin.

#özet alınacak metin

text ="""
Aytu’s Great Escape
As crimson smoke rose from the foothills of the grand mountains, greeting the savage nature of the primeval world, the young and agile hunter Aytu moved silently through the gigantic ferns. The sky was filled with flying reptiles whose wingspans were as wide as a small hut. Relying only on his spear made of bone and sharp stones, Aytu realized he had strayed too far from his tribe's territory for hunting only when he noticed the eerie silence around him. The wind had suddenly died down, and the cheerful chirping of the forest had given way to a deadly stillness.

Suddenly, the ground beneath his feet began to tremble. The water in a nearby pond rippled and shook at regular intervals. Aytu watched as the massive bushes behind him split in two with a deafening crash. Standing before him was a colossal Tyrannosaurus Rex, making the earth shudder with each step, its razor-sharp teeth glistening in the daylight. The monster's yellow, predatory eyes locked directly onto Aytu. The young hunter knew that if he wanted to survive, he had to make a decision within seconds; gripping his spear tightly, he began to run with all his might.

As he fled into the depths of the forest, the massive roar coming from behind was virtually deafening. Aytu used the thick trunks of the trees as shields, running in zigzags, hoping the giant reptile would slow down in tight spaces due to its massive size. The monster, however, continued its relentless pursuit, crushing the huge palm trees and boulders in its path. Its hot breath could almost be felt on the back of Aytu's neck. Just where the path seemed to end, a deep canyon with no visible bottom appeared.

Aytu did not hesitate when he reached the edge of the canyon. Below was a raging river rushing wildly. When he looked back, he saw the drool splattering from the monster's mouth and its massive claw swinging towards him. He didn't have a single second to lose. Before throwing himself into the void, he desperately grabbed onto one of the thick vines hanging from the cliff's edge. Swinging in a wide arc through the air, he released himself into the cool waters of the river.

As he plunged into the water, the strong current began to drag him away rapidly, but this was his salvation. When he surfaced and looked back, he saw the giant dinosaur standing at the edge of the canyon, roaring furiously into the sky. Aytu dragged himself ashore where the river calmed down and took a deep breath. Despite the scratches on his body and his sheer exhaustion, he smiled; he had managed to survive against nature's most dangerous titan using his wits and courage.
"""

"""özet :Eski çağlarda yaşayan çevik avcı Aytu, avlanmak için kabilesinden uzaklaştığında ormanda ölümcül bir sessizlikle karşılaştı.

Toprağın sarsılmasıyla birlikte, karşısında keskin dişleri ve devasa cüssesiyle aç bir Tyrannosaurus Rex belirdi.

Hayatta kalmak için amansızca koşmaya başlayan Aytu, ağaç gövdelerini kalkan olarak kullanarak dev canavarı zikzaklarla atlatmaya çalıştı.

Kaçış yolu derin bir kanyonun kenarında son bulunca, arkasından gelen pençeden kurtulmak için uçurumdaki sarmaşıklara tutunup kendini aşağıdaki nehre bıraktı.

Azgın akıntı sayesinde dev dinazordan uzaklaşmayı başaran Aytu, nehrin kıyısına ulaştığında zekası ve cesaretiyle hayatta kalmanın gururunu yaşadı.
"""

#modeli calıştırıp özetleme işlemini gercekleştirme (eski sürüm)
# summary = summarizer(
#     text,
#     max_length= 100,#özetin maksimum 100 token olmasını sağlar
#     min_lenght = 10,#en az 10 token uzunluğu
#     do_sample = True #rastgelelik ekleyerek modelin her seferinde farklı özetler üretmesini sağlıyor
# )

# print(summary[0]["summary_text"]) 

messages = [
    {
        "role": "user",
        "content": f"Aşağıdaki metni ana fikrini koruyarak kısaca özetle:\n\n{text}"
    }
]

# Özet üretimi
outputs = summarizer(messages, max_new_tokens=150)
summary = outputs[0]["generated_text"][-1]["content"]

print("Özet:")
print(summary)

"""
Özet:
In the heart of the grand mountains, Aytu, an agile hunter,
moves silently through the towering ferns. As the sun sets over the vast landscape,
he notices the eerie silence enveloping the area. The wind has stopped, leaving the ground trembling beneath his feet.
Suddenly, the ground beneath his feet begins to shake, causing the water in a nearby pond to ripple and shake violently. 
Aytu watches as the massive bushes behind him split in two with a deafening crash, revealing a colossal Tyrannosaurus Rex looming overhead,
its razor-sharp teeth gleaming in the light of day. 
"""