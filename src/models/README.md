# Teknofest 2023 - Aşağılayıcı Söylemlerin Doğal Dil İşleme ile Tespiti
### Takım: L2 Regülasyon

---

## 1. Literatür Taraması

Aşağılayıcı söylemleri sınıflandırma görevi için literatürde çeşitli yaklaşımlar öneren birçok yayın bulunmaktadır. Bu problemi araştıran yazarlar yanlılık (bias) problemi üzerine yoğunlaşmış; bu durumun çözümü için farklı ön işleme ve yaklaşımlar uygulamışlardır. Aşağıda literatürdeki çalışmalar listelenmiştir.   

- [Aken B. V., Risch J. Krestel R. ve Löser A., (2018)](https://arxiv.org/abs/1809.07572) `Other, Toxic, Obscene, Insult, Identity Hate, Severe Toxic ve Threat` sınıflarını tahmin etmek için yaptıkları çalışmada en iyi F1 skorunu `Bidirectional GRU Attention (FastText)` modeli ile elde etmişlerdir. Ancak kullandıkları diğer modellerde de skorların benzer olduğu gözlemlenmiştir.
- [Duchêne C., Jamet H., Guillaume P. ve Dehak R., (2023)](https://arxiv.org/abs/2301.11125) `Toxicity, Obscene, Sexual Explict, Identity Attack, Insult ve Threat` sınıflarının tahmini için `BERT, RNN, XLNET` mimarilerini kullanan modeller ile denemeler yapmışlardır ve bütün bu modellerin benzer sonuçlar verdiğini vurgulamışlardır. Nihai olarak; `Focall Loss` ile eğitilmiş `RoBERTa` modeli AUROC ve F1 olarak en iyi sonucu veren model olmuştur.
- [Jhaveri M., Ramaiya D. ve Chadha H. S (2022)](https://arxiv.org/abs/2201.00598) `Abusive ve Not Abuse` sınıflarını birden çok dil (multilingual) için tahmin ederken `BERT` ailesinden 18 farklı model kullanmışlardır. `XLM-RoBERTa Large` en iyi sonucu veren model olmuştur.

## 2. Kullanılan Dış Veriler
Yarışma süresince genellebilir bir model oluşturmak adına hem türkçe hem de diğer dillerden birçok açık veri kaynağını taradık. Kullandığımız veriler aşağıda listelenmiştir.

- [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/data)
- [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [A corpus of Turkish offensive language](https://coltekin.github.io/offensive-turkish/)
- [Turkish Tweets Sentiment Analysis](https://github.com/ezgisubasi/turkish-tweets-sentiment-analysis/blob/main/data/tweetset.csv)
- [Türkçe Sosyal Medya Paylaşımı Veri Seti](https://www.kaggle.com/datasets/mrtbeyz/trke-sosyal-medya-paylam-veri-seti)

Bu verilerin farklı kombinasyonlarını kullanarak ilk olarak pretraining amaçlı base modelimizi eğittik. Sonrasında birinci aşamadan gelen model ağırlıkları (weights) ile yarışma datasında fine-tune ettik. 
TODO: Bu yöntem ile modeldeki yanlılığı azaltmış olsak da yarışma metriğini kötüleştirdiği için kullanmadık. 

## 3. Denenen Modeller

Aşağıda tahminleme süreci boyunca denemiş olduğumuz embedding modeller ve final sınıflandırma katmanları listelenmiştir.

### 3.1. Embedding Model Havuzu
Aşama 1 olarak adlandırabileceğimiz bu kısım, bizlere iletilen `df['text']` sütunundaki metinleri sayısal olarak `N` boyutunda bir vektörde temsil etmemizi sağlamaktadır. Böylece aşama 2 adımında sınıflandırma görevi için modellere öznitelik (feature) sağlayabiliriz. Aşama 1'de oluşturulan vektörler probleme ne kadar uyumlu olursa, sınıflandırma aşaması sonuçlarının da o kadar iyileşmesi beklenmektedir. Bu nedenle farklı mimarileri (architecture) içeren geniş bir havuz oluşturmayı hedefledik ve aşağıdaki gibi listeledik.

#### 3.1.1. TF-IDF 
TF-IDF, bir belgedeki (corpus) her bir kelimenin değerlerini, belirli bir belgedeki kelimenin sıklığı ile kelimenin göründüğü belgelerin yüzdesinin tersiyle hesaplar. Temel olarak TF-IDF, belirli bir belgede kelimelerin göreceli sıklığını, bu kelimenin tüm veri seti üzerindeki tersine oranına göre belirleyerek çalışır. Sezgisel olarak, bu hesaplama, belirli bir kelimenin belirli bir belge ile ne kadar alakalı olduğunu belirler. Tek veya küçük bir belge grubunda ortak olan kelimeler genel kelimelerden daha yüksek TFIDF numaralarına sahip olma eğilimindedir [(Kaynak)](https://dergipark.org.tr/tr/pub/deumffmd/issue/59584/678547).

<p align="center">
  <img src="https://user-images.githubusercontent.com/42123801/229319826-1fedf02c-a2d7-485a-8cb0-782cbdb75b3e.png" width="400"/>
</p>

TF-IDF için hem karakter bazlı hem de kelime bazlı yaklaşımlar denedik. Uygulama yöntemleri için: [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) 

#### 3.1.2. FastText
FastText 2016 yılında Facebook tarafından geliştirilen Word2Vec tabanlı bir modeldir. Bu yöntemin Word2Vec’ten farkı, kelimelerin ngramlara ayrılmasıdır. Böylece Word2Vec ile yakalanamayan anlam yakınlığı bu yöntemle yakalanabilir [(Kaynak)](https://dergipark.org.tr/tr/pub/deumffmd/issue/59584/678547).

<p align="center">
  <img src="https://user-images.githubusercontent.com/42123801/229319555-81cd444f-a478-44ac-85b9-d8cd596e5231.png" width="800"/>
</p>

#### 3.1.3. BERT Backbone

BERT modeli, bir sorguyu ve bir dizi anahtar-değer çiftini bir çıktıya eşlemektedir. Burada sorgu, anahtarlar, değerler ve çıktının kendi aralarındaki korelasyonu ifade edecek vektörler oluşmaktadır. Çıktı, değerlerin ağırlıklı toplamı ile hesaplanmaktadır. Bir değere atanan ağırlık ise sorguya karşılık gelen anahtarla uyumluluk oranı ile hesaplanmaktadır [(Kaynak)](https://dergipark.org.tr/tr/pub/tbbmd/issue/66300/1004781).

<p align="center">
  <img src="https://user-images.githubusercontent.com/42123801/229323836-7ef22f6b-69a6-488c-bee8-7813dded4331.png" width="400"/>
</p>

Daha iyi sonuçlar almak adına, MDZ Digital Library Team ([dbmdz](https://huggingface.co/dbmdz)) tarafınca Türkçe kaynaklarla eğitilmiş BERT modellerine başvurduk. Bu modellerin boyutları kullanılan vocab büyüklüğüne göre değişmektedir ve şu veriler kullanılarak eğitilmiştir:
- [OSCAR corpus](https://oscar-project.org/)
- [OPUS corpora](https://opus.nlpl.eu/)
- [Kemal Oflazer tarafından sağlanan veri seti](https://www.andrew.cmu.edu/user/ko/)
- [Vikipedi](https://tr.wikipedia.org/wiki/Anasayfa)

Model Listesi: 
- [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)
- [dbmdz/bert-base-turkish-uncased](https://huggingface.co/dbmdz/bert-base-turkish-uncased)
- [dbmdz/bert-base-turkish-128k-cased](https://huggingface.co/dbmdz/bert-base-turkish-128k-cased)
- [dbmdz/bert-base-turkish-128k-uncased](https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased)


#### 3.1.4. RoBERTa Backbone
Model Listesi:
- [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
- [xlm-roberta-large](https://huggingface.co/xlm-roberta-large)

#### 3.1.5. Sentence Transformers
Model Listesi:
- [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)


---

#### 3.2. Sınıflandırma Katmanı Havuzu
Lorem ipsum

#### 3.2.1. LightGBM
Lorem ipsum

#### 3.2.2. XGBoost
Lorem ipsum

#### 3.2.3. CatBoost
Lorem ipsum

#### 3.2.4. Support Vector CLassifier (SVC)
Lorem ipsum

#### 3.2.5. Neural SoftMax Katmanı
Lorem ipsum

---

### 4. Modellerin Sonuçları

Lorem ipsum

| Backbone | Classifier           | Macro-F1 |
|--------|--------------------| ---------|
| A      | A | 0.9 |
| B      | A  | 0.9  |
