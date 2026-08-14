"""
kücük bir veri seti üzerinden word embedding sonrasında PCA ile görselliştircez
-word2vec : google
-fasttext : meta

1-örnek bir veri seti oluşturuaz
2-preprocessing: cümleleri tokena cevirip
3-word2vec ve fasttext modellerini tanımlıcaz
4-iki modelden elge edilen vektörleri PCA ile 3 boyuta indirgicez
5-kelimeler vektörlerinin 3 boyutlu görselleştirme

pip install pandas matplotlib scikit-learn gensim
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec , FastText
from gensim.utils import simple_preprocess
import numpy as np
#örnek veri seti oluşturucaz
# cumleler = [
#     "köpek cok tatlı bir hayvandır.",
#     "köpekler evcil hayvanlardır.",
#     "kediler genellikle bağımsız hareket etmeyi severler.",
#     "köpekler sadık ve dost canlısı hayvanlardır.",
#     "hayvanlar insanlar icin iyi arkadaşlardır.",
#     "Türkiye'nin başkenti Ankaradır.",
#     "Türkiye'de Ankara ve Gaziantep'in yemekleri cok güzel."
# ]

#ödev cümleler
camleler = [
    # Animals (1-100)
    "The loyal dog barked at the stranger near the gate.",
    "A small kitten chased a ball of wool across the living room.",
    "The wild wolf howled at the full moon in the dark forest.",
    "Lions hunt in prides to capture large prey on the savanna.",
    "A graceful deer leaped over the wooden fence effortlessly.",
    "The brown bear caught a fresh salmon from the rushing river.",
    "Elephants are known for their high intelligence and strong memory.",
    "A sneaky fox slid quietly through the thick bushes.",
    "The colorful parrot repeated every word spoken by its owner.",
    "Dolphins swim gracefully alongside the fast-moving ocean ships.",
    "A huge blue whale surfaced to breathe fresh air.",
    "The slow turtle crawled across the hot dusty road.",
    "Horses run fast across the open green pastures.",
    "A fierce tiger stalked its target through the jungle.",
    "Rabbits love to dig deep burrows under the soil.",
    "The wise owl sat on a tall tree branch all night.",
    "A sharp eagle flew high above the snow-capped mountains.",
    "Monkeys swing effortlessly between tree branches using their tails.",
    "The pink flamingo stood on one leg in the shallow lake.",
    "A dangerous shark swam silently below the ocean surface.",
    "The busy bee collected sweet nectar from bright flowers.",
    "Ants work together in large organized colonies under the ground.",
    "A green frog jumped into the cool muddy pond water.",
    "The spotted leopard climbed the tall tree with ease.",
    "Kangaroos hop quickly across the dry Australian outback.",
    "A heavy hippopotamus rested in the cool river mud.",
    "The tall giraffe reached for the highest tree leaves.",
    "A tiny hummingbird flapped its wings extremely fast.",
    "The black crow rested on top of the telephone wire.",
    "Sheep graze peaceful on the green hills all day long.",
    "The noisy rooster crowed early in the morning sun.",
    "Ducks swam gently across the quiet neighborhood park lake.",
    "A giant panda munched on fresh green bamboo shoots.",
    "The lazy cat slept on the warm sunny windowsill.",
    "Wolves communicate with each other using loud howls.",
    "A sea turtle swims thousands of miles across the ocean.",
    "The clever chimpanzee used a stick to catch insects.",
    "Cheetahs are the fastest land animals in the entire world.",
    "A polar bear walked carefully across the white icy surface.",
    "The golden retriever retrieved the thrown wooden stick quickly.",
    "A striped zebra ran with the herd across the dry plain.",
    "Owls hunt small rodents during the darkest hours of night.",
    "A poisonous snake slithered through the long dry grass.",
    "The gray squirrel buried acorns deep inside the soil.",
    "A beautiful butterfly fluttered around the blooming garden flowers.",
    "Sea otters float on their backs while floating in water.",
    "The strong gorilla guarded its family inside the dense forest.",
    "A black panther moved like a shadow in the night.",
    "The noisy seagull flew over the ocean beach looking for food.",
    "Camel herds travel long distances through the hot desert sand.",
    "A tiny mouse hid inside a small hole in the wall.",
    "The majestic falcon dived fast from the high sky.",
    "A giant octopus hid inside a deep ocean cave.",
    "The farm pig splashed happily in the cool mud pool.",
    "Swans glided smoothly across the clear mirror-like water.",
    "A wild boar searched for roots beneath the damp forest floor.",
    "The sleepy sloth hung upside down from the sturdy branch.",
    "A hungry alligator waited motionless in the dark swamp water.",
    "Flocks of wild geese fly south during the cold winter.",
    "The playful puppy chewed on a rubber ball excitedly.",
    "A badger dug a deep burrow in the side of the hill.",
    "The sea lion rested comfortably on the flat sunny rock.",
    "A sharp hawk spotted its target from high above the field.",
    "The koala rested quietly while eating sweet eucalyptus leaves.",
    "A white goat climbed the steep rocky mountain side easily.",
    "The blue jay sang a loud song from the top of the pine tree.",
    "Porcupines protect themselves using sharp quills on their backs.",
    "A beaver built a strong dam using fallen tree logs.",
    "The walrus rested on floating ice blocks in the arctic ocean.",
    "A green iguana basked on the sunny rock all afternoon.",
    "The energetic terrier chased after the small stray tennis ball.",
    "A pelican caught a large fish in its giant beak pouch.",
    "The wild stallion ran freely across the wide grassy plain.",
    "A tiny ladybug crawled slowly across the green leaf surface.",
    "The grizzly bear hunted for fresh berries in the mountain valley.",
    "A chameleon changed its skin color to match the tree trunk.",
    "The sea anemone floated gently in the warm tide pool.",
    "A flock of pigeons flew together over the crowded city square.",
    "The arctic fox changed its fur to pure white in winter.",
    "A lobster crawled along the rocky bottom of the cold ocean.",
    "The wild hyena laughed loudly in the dark night air.",
    "A tiny hamster ran on its plastic exercise wheel endlessly.",
    "The spotted hyena searched for food left by the lions.",
    "A pelican glided low right above the ocean ocean waves.",
    "The woodchucks dug tunnels deep underneath the grassy lawn.",
    "A blue whale sang deep ocean songs across vast distances.",
    "The stray cat drank warm milk from a small clean saucer.",
    "A wild turkey walked slowly across the dusty country road.",
    "The sea urchin rested on the ocean floor near the reef.",
    "A bright red cardinal perched on the snow-covered branch.",
    "The farm cow stood quietly while eating fresh green hay.",
    "A swift swallow built its nest out of river mud.",
    "The giant tortoise walked very slowly across the dry ground.",
    "A wild bison herd moved across the vast open prairie.",
    "The night owl turned its head around to look for food.",
    "A jellyfish floated silently through the deep blue ocean.",
    "The black raven perched on top of the old wooden fence post.",
    "A small hedgehog curled into a spiky ball for defense.",
    "The ocean crab walked sideways along the wet sandy shore.",
    "A young deer drank cold water from the clear forest stream.",

    # Sports (101-200)
    "The striker kicked the football straight into the top corner of the net.",
    "The basketball player scored a sensational three-pointer in the final second.",
    "The tennis player served an powerful ace down the middle line.",
    "Athletes lined up at the starting line for the hundred meter sprint.",
    "The goalkeeper made a miraculous dive to save the penalty shot.",
    "The swimmer dove into the lane and swam forty meters quickly.",
    "The baseball batter hit a massive home run out of the stadium.",
    "The boxer delivered a fast jab to win the championship belt.",
    "The marathon runner crossed the finish line after forty-two kilometers.",
    "The referee blew the whistle to signal the end of the match.",
    "The volleyball team celebrated after a brilliant jump block at the net.",
    "The golfer sunk a difficult ten-foot putt on the final green.",
    "The rugby player tackled the opponent before he reached the try line.",
    "The gymnast performed an amazing backflip on the balance beam.",
    "The cyclist accelerated past the group on the mountain climb.",
    "The hockey player slapped the puck hard past the goalkeeper.",
    "The wrestler pinned his opponent down to win the match.",
    "The skier glided down the steep snowy slope at high speed.",
    "The surfer rode a massive ocean wave all the way to the beach.",
    "The archer shot an arrow directly into the center bullseye.",
    "The team captain lifted the championship trophy high in the air.",
    "The coach called a quick timeout to adjust the defensive tactic.",
    "The midfielder passed the ball across the field to the open winger.",
    "The runner trained every single morning on the local track field.",
    "The badminton player executed a powerful smash across the net.",
    "The weightlifter lifted two hundred kilograms above his head cleanly.",
    "The figure skater landed a perfect triple axel on the cold ice.",
    "The cricket batsman scored a brilliant century in front of the home crowd.",
    "The sprinter broke the national world record by two milliseconds.",
    "The basketball center grabbed an important offensive rebound under the hoop.",
    "The quarterback threw a fifty-yard touchdown pass to the receiver.",
    "The rower pulled hard on the oars to lead the river race.",
    "The table tennis player returned the fast ping-pong ball effortlessly.",
    "The soccer defender tackled the ball cleanly near the penalty box.",
    "The Formula One driver sped past the finish line to take victory.",
    "The rock climber reached the rocky summit after hours of hard climbing.",
    "The tennis star played a magnificent drop shot over the net.",
    "The baseball pitcher threw a fast ninety-five mile per hour curveball.",
    "The marathon was postponed due to heavy rain and strong wind.",
    "The high jumper cleared the bar set at two meters high.",
    "The swimmer won four gold medals in the international championship competition.",
    "The football coach instructed the squad during the half-time break.",
    "The skateboarder landed a clean kickflip off the staircase ledge.",
    "The ice hockey goalie caught the flying puck with his padded glove.",
    "The boxer ducked under the punch and countered with a strong hook.",
    "The golf tournament attracted thousands of spectators from all over.",
    "The pole vaulter soared gracefully over the high wooden bar.",
    "The runner wore specialized lightweight shoes for the marathon.",
    "The referee handed out a red card for a dangerous high tackle.",
    "The point guard dribbled the basketball smoothly through the defense.",
    "The baseball catcher caught the third strike to end the inning.",
    "The competitive swimmer improved her personal best lap time today.",
    "The karate master executed a swift front kick to the target pad.",
    "The javelin thrower tossed the spear deep into the grassy field.",
    "The stadium cheered loudly when the local team scored a goal.",
    "The cyclist wore a helmet for safety during the fast road race.",
    "The volleyball passer set the ball high for the outside hitter.",
    "The snooker player potted the final black ball to win the frame.",
    "The long jumper sprinted down the track and leaped into the sand pit.",
    "The soccer match ended in a thrilling two-two draw after extra time.",
    "The oarsmen rowed in perfect sync along the calm river water.",
    "The basketball team used a full-court press to force a turnover.",
    "The skiing instructor showed the beginners how to stop safely on snow.",
    "The tennis opponent challenged the referee call with video review.",
    "The marathon route went through the historic streets of the city center.",
    "The weightlifter chalked his hands before gripping the heavy metal bar.",
    "The boxer trained hard with the heavy punching bag every single afternoon.",
    "The football match was full of intense tackles and fast counter-attacks.",
    "The diver executed a triple somersault before entering the pool water.",
    "The field hockey player dribbled past three defenders with precise stick control.",
    "The sprinter leaned forward at the line to win the close race.",
    "The basketball coach emphasized strong defensive positioning in practice.",
    "The golf ball rolled slowly down the slope and into the small cup.",
    "The surf competition was held during the massive summer swell.",
    "The baseball team loaded the bases with three consecutive hits.",
    "The runner maintained a steady pace throughout the long distance race.",
    "The ice skater twirled rapidly on one foot in the center of the rink.",
    "The goalkeeper punched the dangerous corner kick far out of the area.",
    "The archery competitor aimed carefully at the target in the quiet arena.",
    "The sports fans waved large flags and sang songs in the stands.",
    "The badminton shuttlecock flew fast across the high indoor court.",
    "The mountain biker navigated the rough trail filled with big rocks.",
    "The rugby team won the scrum and pushed the opposition backwards.",
    "The athlete warmed up with gentle stretching before the big race.",
    "The tennis champion lifted the grand slam trophy in front of millions.",
    "The swimming pool was heated to the perfect official competition temperature.",
    "The soccer player received a yellow card for pulling the opponent shirt.",
    "The bench players stood up and cheered when their team scored a point.",
    "The marathon water station helped runners stay hydrated along the course.",
    "The gymnast landed on the mat with complete control and balance.",
    "The Formula One pit crew changed all four tires in under two seconds.",
    "The basketball shooter made five consecutive shots from the corner.",
    "The volleyball player served a powerful jump serve over the white net.",
    "The chess player executed a clever tactical sacrifice on the board.",
    "The baseball outfielder caught the high fly ball near the stadium wall.",
    "The runner sprinted past his rival in the final fifty meters.",
    "The football team practiced short passing drills under the hot afternoon sun.",
    "The mountain climber used heavy ropes to scale the freezing ice wall.",
    "The swimmer touched the wall first to set a new national record.",
    "The championship game concluded with an amazing victory celebration on the field."
]
cumleler = [
    # Hayvan cümleleri
    "the dog barked at the stranger",
    "a cat was sleeping on the sofa",
    "the lion is the king of the jungle",
    "the tiger is hunting in the forest",
    "a big bear was caught fishing",
    "the owl flies quietly at night",
    "a shark swims in the ocean",
    "the whale is a huge sea animal",
    "an eagle flies high in the sky",
    "the wolf pack howling together",
    
    # Spor cümleleri
    "they played a game of football",
    "he scored a goal in the match",
    "the basketball player jumped high",
    "she played tennis on the court",
    "the soccer team won the match",
    "the boxer trains hard for the fight",
    "the swimmer broke the record in the pool",
    "the runner finished the race fast"
]

tokenize_cumleler = [simple_preprocess(c) for c in cumleler]
print(tokenize_cumleler)
# """
# [['köpek', 'cok', 'tatlı', 'bir', 'hayvandır'], ['köpekler', 'evcil', 'hayvanlardır'], ['kediler', 'genellikle', 'bağımsız', 'hareket', 'etmeyi', 'severler'], ['köpekler', 'sadık', 've', 'dost', 'canlısı', 'hayvanlardır'], ['hayvanlar', 'insanlar', 'icin', 'iyi', 'arkadaşlardır'], ['türkiye', 'nin', 'başkenti', 'ankaradır'], ['türkiye', 'de', 'ankara', 've', 'gaziantep', 'in', 'yemekleri', 'cok', 'güzel']]
# """

# Word2Vec_model = Word2Vec(
#     sentences = tokenize_cumleler,
#     vector_size=10 ,
#     window=5,
#     min_count=1,#en az kac defa gecen cümleler var
#     sg=0,
#     epochs=50
# )
yeni_model = Word2Vec(
    sentences = tokenize_cumleler,
    vector_size=10 ,
    window=5,
    min_count=1,#en az kac defa gecen cümleler var
    sg=0,
    epochs=50
)
fasttext_model = FastText(
    sentences = tokenize_cumleler,
    vector_size=50,
    window=5,
    min_count=1,
    sg=0
)
def plot_word_embeddings(model,baslik):
    #kelime vektörleri alma
    kelime_vektor = model.wv

    #ilk 1000 kelime alma
    kelimeler =list(kelime_vektor.index_to_key)[:1000]
    vektorler = [kelime_vektor[w] for w in kelimeler]

    #PCA ile boyut indirgemesi
    pca=PCA(n_components=3)
    indirgenmis_vektorler = pca.fit_transform(vektorler)

    #3d görselleştirme
    fig = plt.figure(figsize= (8,6))
    ax = fig.add_subplot(111, projection = "3d")

    #noktaları cizdirme
    ax.scatter(indirgenmis_vektorler[:,0], indirgenmis_vektorler[:,1],indirgenmis_vektorler[:,2] )

    for i , kelime in enumerate(kelimeler):
        ax.text(indirgenmis_vektorler[i,0], indirgenmis_vektorler[i,1],indirgenmis_vektorler[i,2] , kelime , fontsize =11)

    ax.set_title(baslik)
    ax.set_xlabel("bileşen 1")
    ax.set_ylabel("bileşen 2")
    ax.set_zlabel("bileşen 3")
    plt.show()
secilen_kelimeler = [
    # Hayvanlar
    "dog", "cat", "lion", "tiger", "bear", "owl", "shark", "whale", "eagle", "wolf",
    # Sporlar
    "football", "basketball", "tennis", "soccer", "boxer", "swimmer", "runner"
]
def plot_custom_words(model, word_list):
    # Sadece listede olan ve modelde bulunan kelimelerin vektörlerini alma
    valid_words = [w for w in word_list if w in model.wv]
    vectors = np.array([model.wv[w] for w in valid_words])
    
    # Boyut indirgeme (PCA)
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(vectors)
    
    # 3D Çizim
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, word in enumerate(valid_words):
        ax.scatter(reduced[i, 0], reduced[i, 1], reduced[i, 2])
        ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2], word, fontsize=10)
        
    ax.set_title("Temizlenmiş Word2Vec Uzayı")
    plt.show()
plot_custom_words(yeni_model, secilen_kelimeler)
# plot_word_embeddings(yeni_model,"Word2Vec Gösterimi")

# plot_word_embeddings(fasttext_model , "Fasttext Gösterimi")

# Hayvan - Hayvan benzerliği
print("Dog & Wolf benzerliği:", yeni_model.wv.similarity("dog", "wolf"))

# Hayvan - Spor benzerliği
print("Dog & Basketball benzerliği:", yeni_model.wv.similarity("dog", "basketball"))