# Teknofest 2023 - Aşağılayıcı Söylemlerin Doğal Dil İşleme ile Tespiti
### Takım: L2 Regülasyon

---

### Literatür Taraması

Aşağılayıcı söylemleri sınıflandırma görevi için literatürde çeşitli yaklaşımlar öneren birçok yayın bulunmaktadır. Bu problemi araştıran yazarlar yanlılık (bias) problemi üzerine yoğunlaşmış; bu durumun çözümü için farklı ön işleme ve yaklaşımlar uygulamışlardır. Aşağıda literatürdeki çalışmalar listelenmiştir.   

* [Aken B. V., Risch J. Krestel R. ve Löser A., 2018](https://arxiv.org/abs/1809.07572) `Other, Toxic, Obscene, Insult, Identity Hate, Severe Toxic ve Threat` sınıflarını tahmin etmek için yaptıkları çalışmada en iyi F1 skorunu `Bidirectional GRU Attention (FastText)` modeli ile elde etmişlerdir. Ancak kullandıkları diğer modellerde de skorların benzer olduğu gözlemlenmiştir.

* [Duchêne C., Jamet H., Guillaume P. ve Dehak R., 2023](https://arxiv.org/abs/2301.11125) `Toxicity, Obscene, Sexual Explict, Identity Attack, Insult ve Threat` sınıflarının tahmini için `BERT, RNN, XLNET` altyapılarını kullanan modeller ile denemeler yapmışlardır ve bütün bu modellerin benzer sonuçlar verdiğini vurgulamışlardır. Nihai olarak; `Focall Loss` ile eğitilmiş `RoBERTa` modeli AUROC ve F1 olarak en iyi sonucu veren model olmuştur.

* [Jhaveri M., Ramaiya D. ve Chadha H. S (2022)](https://arxiv.org/abs/2201.00598) `Abusive ve Not Abuse` sınıflarını birden çok dil (multilingual) için tahmin ederken `BERT` ailesinden 18 farklı model kullanmışlardır. `XLM-RoBERTa Large` en iyi sonucu veren model olmuştur.

### Kullanılan Dış Veriler
Yarışma süresince genellebilir bir model oluşturmak adına hem türkçe hem de diğer dillerden birçok açık veri kaynağını taradık. Kullandığımız veriler aşağıda listelenmiştir.

* [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/data)
* [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
* [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
* [A corpus of Turkish offensive language](https://coltekin.github.io/offensive-turkish/)
* [Turkish Tweets Sentiment Analysis](https://github.com/ezgisubasi/turkish-tweets-sentiment-analysis/blob/main/data/tweetset.csv)
* [Türkçe Sosyal Medya Paylaşımı Veri Seti](https://www.kaggle.com/datasets/mrtbeyz/trke-sosyal-medya-paylam-veri-seti)

Bu verilerin farklı kombinasyonlarını kullanarak ilk olarak pretraining amaçlı base modelimizi eğittik. Sonrasında birinci aşamadan gelen model ağırlıkları (weights) ile yarışma datasında fine-tune ettik. 
TODO: Bu yöntem ile modeldeki yanlılığı azaltmış olsak da yarışma metriğini kötüleştirdiği için kullanmadık. 

### Denenen Modeller

Aşağıda tahminleme süreci boyunca denemiş olduğumuz embedding modeller ve final sınıflandırma katmanları listelenmiştir.

#### 1. Embedding Model Havuzu
Lorem ipsum

#### 1.1. TFIDF
Lorem ipsum

#### 1.2. FastText
Lorem ipsum

#### 1.3. BERT Backbone
Model listesi:
- [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)
- [dbmdz/bert-base-turkish-uncased](https://huggingface.co/dbmdz/bert-base-turkish-uncased)

Lorem ipsum

#### 1.4. RoBERTa Backbone
Model listesi:
- [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
- [xlm-roberta-large](xlm-roberta-large)

Lorem ipsum

---

#### 2. Sınıflandırma Katmanı Havuzu
Lorem ipsum

#### 1.1. LightGBM
Lorem ipsum

#### 1.2. XGBoost
Lorem ipsum

#### 1.3. CatBoost
Lorem ipsum

#### 1.4. Support Vector CLassifier (SVC)
Lorem ipsum

#### 1.5. Neural SoftMax Katmanı
Lorem ipsum

---

### Modellerin Sonuçları

Lorem ipsum

| Mimari | Macro-F1           |
|--------|--------------------|
| A      | 0.9 |
| B      | 0.9  |
