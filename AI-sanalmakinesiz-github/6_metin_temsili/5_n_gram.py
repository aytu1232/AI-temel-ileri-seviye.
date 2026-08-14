from sklearn.feature_extraction.text import CountVectorizer

#örnek belge

belgeler = [
    "Bu calışma bir NGRAM calışmasıdır.",
    "bu calışma doğal dil işleme calışmasıdır."
]

unigram_model = CountVectorizer(ngram_range=(1,1))

bigram_model = CountVectorizer(ngram_range=(2,2))

trigram_model = CountVectorizer(ngram_range=(3,3))

X_unigram = unigram_model.fit_transform(belgeler)

unigram_ozellikler = unigram_model.get_feature_names_out()
###################################################################
X_bigram = bigram_model.fit_transform(belgeler)

bigram_ozellikler = bigram_model.get_feature_names_out()
#################################################################
X_trigram = trigram_model.fit_transform(belgeler)

trigram_ozellikler = trigram_model.get_feature_names_out()


print(f"Unigram : {unigram_ozellikler}\n bigram : {bigram_ozellikler}\n trigram :{trigram_ozellikler}")

"""
Unigram : ['bir' 'bu' 'calışma' 'calışmasıdır' 'dil' 'doğal' 'işleme' 'ngram']
 bigram : ['bir ngram' 'bu calışma' 'calışma bir' 'calışma doğal' 'dil işleme'
 'doğal dil' 'işleme calışmasıdır' 'ngram calışmasıdır']
 trigram :['bir ngram calışmasıdır' 'bu calışma bir' 'bu calışma doğal'
 'calışma bir ngram' 'calışma doğal dil' 'dil işleme calışmasıdır'
 'doğal dil işleme']
 """