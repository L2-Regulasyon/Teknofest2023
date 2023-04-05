# Teknofest 2023 - Aşağılayıcı Söylemlerin Doğal Dil İşleme ile Tespiti
### Takım: L2 Regülasyon

---

### Denenen Modeller / Yaklaşımlar ve Sonuçları

---

## 1. Literatür Taraması

Aşağılayıcı söylemleri sınıflandırma görevi için literatürde çeşitli yaklaşımlar öneren birçok yayın bulunmaktadır. Bu problemi araştıran yazarlar yanlılık (bias) problemi üzerine yoğunlaşmış; bu durumun çözümü için farklı ön işleme ve yaklaşımlar uygulamışlardır. Aşağıda literatürdeki çalışmalar listelenmiştir.


- [Aken B. V., Risch J. Krestel R. ve Löser A., (2018)](https://arxiv.org/abs/1809.07572) `Other, Toxic, Obscene, Insult, Identity Hate, Severe Toxic` ve `Threat` sınıflarını tahmin etmek için yaptıkları çalışmada en iyi F1 skorunu `Bidirectional GRU Attention (FastText)` modeli ile elde etmişlerdir. Ancak kullandıkları diğer modellerde de skorların benzer olduğu gözlemlenmiştir.
- [Duchêne C., Jamet H., Guillaume P. ve Dehak R., (2023)](https://arxiv.org/abs/2301.11125) `Toxicity, Obscene, Sexual Explict, Identity Attack, Insult` ve `Threat` sınıflarının tahmini için `BERT, RNN, XLNET` mimarilerini kullanan modeller ile denemeler yapmışlardır ve bütün bu modellerin benzer sonuçlar verdiğini vurgulamışlardır. Nihai olarak; `Focall Loss` ile eğitilmiş `RoBERTa` modeli AUROC ve F1 olarak en iyi sonucu veren model olmuştur.
- [Jhaveri M., Ramaiya D. ve Chadha H. S (2022)](https://arxiv.org/abs/2201.00598) `Abusive` ve `Not Abuse` sınıflarını birden çok dil (multilingual) için tahmin ederken `BERT` ailesinden 18 farklı model kullanmışlardır. `xlm-roberta-large` en iyi sonucu veren model olmuştur.

## 2. Kullanılan Dış Veriler
Yarışma süresince genellebilir bir model oluşturmak adına hem türkçe hem de diğer dillerden birçok açık veri kaynağını taradık. Kullandığımız veriler aşağıda listelenmiştir.

- [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/data)
- [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [A corpus of Turkish offensive language](https://coltekin.github.io/offensive-turkish/)
- [Turkish Tweets Sentiment Analysis](https://github.com/ezgisubasi/turkish-tweets-sentiment-analysis/blob/main/data/tweetset.csv)
- [Türkçe Sosyal Medya Paylaşımı Veri Seti](https://www.kaggle.com/datasets/mrtbeyz/trke-sosyal-medya-paylam-veri-seti)

Bu verilerin farklı kombinasyonlarını kullanarak ilk olarak pretraining amaçlı base modelimizi eğittik. Sonrasında birinci aşamadan gelen model ağırlıkları (weights) ile yarışma datasında fine-tune ettik. Bu yöntem ile modeldeki yanlılığı azalttık ve production ortamı için genellenebilir bir model oluşturduk.

## 3. Denenen Modeller

Aşağıda tahminleme süreci boyunca denemiş olduğumuz embedding modeller ve final sınıflandırma katmanları listelenmiştir.

### 3.1. Embedding Model Havuzu
Aşama 1 olarak adlandırabileceğimiz bu kısım, bizlere iletilen `df['text']` sütunundaki metinleri sayısal olarak `N` boyutunda bir vektörde temsil etmemizi sağlamaktadır. Böylece AŞama 2'de sınıflandırma görevi için modellere öznitelik (feature) sağlayabiliriz. Aşama 1'de oluşturulan vektörler probleme ne kadar uyumlu olursa, sınıflandırma aşaması sonuçlarının da o kadar iyileşmesi beklenmektedir. Bu nedenle farklı mimarileri (architecture) içeren geniş bir havuz oluşturmayı hedefledik ve aşağıdaki gibi listeledik.

#### 3.1.1. TF-IDF - [Referans](https://dergipark.org.tr/tr/pub/deumffmd/issue/59584/678547)
TF-IDF, bir belgedeki (corpus) her bir kelimenin değerlerini, belirli bir belgedeki kelimenin sıklığı ile kelimenin göründüğü belgelerin yüzdesinin tersiyle hesaplar. Temel olarak TF-IDF, belirli bir belgede kelimelerin göreceli sıklığını, bu kelimenin tüm veri seti üzerindeki tersine oranına göre belirleyerek çalışır. Sezgisel olarak, bu hesaplama, belirli bir kelimenin belirli bir belge ile ne kadar alakalı olduğunu belirler. Tek veya küçük bir belge grubunda ortak olan kelimeler genel kelimelerden daha yüksek TFIDF numaralarına sahip olma eğilimindedir.

<p align="center">
  <img src="https://user-images.githubusercontent.com/42123801/229319826-1fedf02c-a2d7-485a-8cb0-782cbdb75b3e.png" width="400"/>
</p>

TF-IDF için hem karakter bazlı hem de kelime bazlı yaklaşımlar denedik. Uygulama yöntemleri için: [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) 

#### 3.1.2. FastText - [Referans](https://dergipark.org.tr/tr/pub/deumffmd/issue/59584/678547)
FastText 2016 yılında Facebook tarafından geliştirilen Word2Vec tabanlı bir modeldir. Bu yöntemin Word2Vec’ten farkı, kelimelerin ngramlara ayrılmasıdır. Böylece Word2Vec ile yakalanamayan anlam yakınlığı bu yöntemle yakalanabilir.

<p align="center">
  <img src="https://user-images.githubusercontent.com/42123801/229319555-81cd444f-a478-44ac-85b9-d8cd596e5231.png" width="800"/>
</p>

FastText için `skipgram` ve `cbow` mimarilerini denedik. Uygulama yöntemler için: [train_unsupervised](https://fasttext.cc/docs/en/python-module.html)

#### 3.1.3. BERT Backbone - [Referans](https://dergipark.org.tr/tr/pub/tbbmd/issue/66300/1004781)

BERT modeli, bir sorguyu ve bir dizi anahtar-değer çiftini bir çıktıya eşlemektedir. Burada sorgu, anahtarlar, değerler ve çıktının kendi aralarındaki korelasyonu ifade edecek vektörler oluşmaktadır. Çıktı, değerlerin ağırlıklı toplamı ile hesaplanmaktadır. Bir değere atanan ağırlık ise sorguya karşılık gelen anahtarla uyumluluk oranı ile hesaplanmaktadır.

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


#### 3.1.4. RoBERTa Backbone - [Referans](https://arxiv.org/abs/1907.11692)

roBERTa (Robustly Optimized BERT pre-training Approach) modellerinin BERT modellerinden ayrıştığı nokta maskingdir. BERT veri hazırlanma aşamasında yalnızca bir kere, statik bir masking yöntemi kullanırken; ROBERTA her bir epoch'da dynamic masking yapmaktadır ve bu nedenle robust olarak atfedilmektedir.

Model Listesi:
- [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
- [xlm-roberta-large](https://huggingface.co/xlm-roberta-large)

#### 3.1.5. Sentence Transformers Backbone - [Referans](https://arxiv.org/abs/1908.10084)

Kullandığımız Sentence Transformerların hepsi BERT ailesine aittir. Sentence Transformerslar genellikle metin ve görsellerin belirli bir vektör uzayında benzerliklerini daha hızlı tespit edebilmek adına geliştirilmiş ve daha çok unsupervised görevlerde kullanılmaktadırlar.

Model Listesi:
- [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

### 3.2. Sınıflandırma Katmanı Havuzu

Bu kısımda Aşama 1'de elde edilen vektörler/öznitelikler kullanılarak farklı mimarilerle sınıflandırma görevi gerçekleştirilmiştir. 

#### 3.2.1. LightGBM - [Referans](https://lightgbm.readthedocs.io/en/v3.3.2/Features.html)

LightGBM, histogram tabanlı çalışan bir boosting (ensemble) yöntemidir. Sürekli değerleri kesikli formata dönüştürerek hesaplama gücü gereksinimi azaltır ve hızı artırır.

LightGBM leaf-wise bölünme yöntemini kullanmaktadır:

<p align="center">
  <img src="https://user-images.githubusercontent.com/42123801/229350911-5bf5f795-e591-4cad-b945-f7f96decf111.png" width="700"/>
</p>

Leaf-wise yaklaşım veriseti küçük olduğunda overfit riski doğurmaktadır ancak doğru parametre seti ile bu tür riskler ortadan kaldırılabilir.

#### 3.2.2. XGBoost  - [Referans](https://xgboost.readthedocs.io/en/stable/)
XGBoost'da (Extreme Gradient Boosting) decison-tree temelli ve gradient-boosting yöntemlerinden biridir. LightGBM'den farklı olarak level-wise yaklaşımı izlemektedir:

<p align="center">
  <img src="https://user-images.githubusercontent.com/42123801/229359120-e3f1fb88-ad48-41be-8117-e1f730d17baf.png" width="700"/>
</p>


#### 3.2.3. CatBoost - [Referans](https://catboost.ai/news/catboost-enables-fast-gradient-boosting-on-decision-trees-using-gpus)

Catboost diğer Gradient Boosting algoritmalarından farklı olarak symmetric tree yöntemini izler:

<p align="center">
  <img src="https://user-images.githubusercontent.com/42123801/229360226-edcf5dd6-5cf8-4d30-99a7-3f54edd5fdd4.png" width="600"/>
</p>

Ayrıca kategorik öznitelikleri daha farklı ele alarak one-hot-encoding dışına çıkar, farklı kategorik değerleri birleştirir ve daha iyi performans gösterir.

#### 3.2.4. Support Vector Classifier (SVC) - [Referans](https://scikit-learn.org/stable/modules/svm.html#multi-class-classification)

Support Vector Machines (SVMs) sınıflandırma, regresyon ve aykırı değerlerin tespiti için kullanılan bir dizi denetimli öğrenme yöntemidir. Vektör boyutu fazla olduğunda avantaj sağlayan bir yöntemdir.


<p align="center">
  <img src="https://user-images.githubusercontent.com/42123801/229606614-865b08f0-0e1c-4631-9fc9-445306e8048c.jpg" width="500"/>
</p>


Multi-class sınıflandırma için ise `one-versus-one` yöntemi izlenerek tahminler oluşturulmaktadır.

#### 3.2.5. Neural SoftMax Katmanı - [Referans](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)

Fine-tune ettiğimiz dil modellerinin son katmanına softmax yerleştirerek her bir class için olasılık dönmesini sağladık.

<p align="center">
  <img src="https://user-images.githubusercontent.com/42123801/229606143-46162f66-dab5-4490-80b2-c4ed96e84438.jpg" width="400"/>
</p>

Böylece Aşama 2 aslında Aşama 1'in içinde yer almış oldu ve nihai çözümümüzde de hem mimari olarak kolaylık sağlamasından hem de başarısından ötürü bu yöntemi kullandık.

---

### 4. Modellerin Sonuçları

Yarışma boyunca birçok mimari ve yöntemi kombinasyonlarıyla denendi. Denemelerimize ve sonuçlarına aşağıdaki tabloda yer verilmiştir. NLP modellerinde hem ayrıklığı azaltmak hem de transformer modellerinin aynı boyutta sözlük dağarcığıyla daha fazla farklı kelimeyi temsil edebilmeleri adına küçük-harf dönüşümü kullanılmıştır. Verilmiş bütün mimari kombinasyonlar aynı split stratejisiyle **RTX4090** üzerinde eğitilmiştir. **Stratified 10 Fold** ve **OOF (Out-of-Fold)** sonuçları raporlanmıştır.

|Model|F1-Macro|F1-OTHER|F1-INSULT|F1-RACIST|F1-SEXIST|F1-PROFANITY|Ortalama Fold Eğitim Süresi|
|---|---|---|---|---|---|---|---|
|toxic-dbmdz-bert-base-turkish-128k-uncased|95.58	|96.63	|92.16	|96.67	|96.43	|95.99	|64.02 +- 0.4s|
|dbmdz-bert-base-turkish-128k-uncased (Fine-Tuned) Embeddings + svc|95.54	|96.59	|92.14	|96.71	|96.28	|95.98	|77.96 +- 0.53s|
|dbmdz-bert-base-turkish-128k-uncased (Fine-Tuned) Embeddings + lgbm|95.5	|96.62	|91.94	|96.6	|96.34	|96.01	|80.71 +- 0.42s|
|dbmdz-bert-base-turkish-128k-uncased (Fine-Tuned) Embeddings + xgb|95.48	|96.59	|91.94	|96.52	|96.41	|95.95	|76.52 +- 0.33s|
|dbmdz-bert-base-turkish-128k-uncased (Fine-Tuned) Embeddings + catboost|95.44	|96.51	|91.91	|96.69	|96.15	|95.95	|81.57 +- 0.31s|
|xlm-roberta-base (Fine-Tuned) Embeddings + lgbm|92.92	|94.35	|87.27	|94.37	|94.66	|93.96	|102.41 +- 0.32s|
|xlm-roberta-base (Fine-Tuned) Embeddings + svc|92.89	|94.24	|87.43	|94.31	|94.48	|93.97	|97.83 +- 0.5s|
|xlm-roberta-base (Fine-Tuned) Embeddings + xgb|92.84	|94.29	|87.21	|94.38	|94.46	|93.87	|97.88 +- 0.31s|
|xlm-roberta-base (Fine-Tuned) Embeddings + catboost|92.84	|94.19	|87.34	|94.48	|94.28	|93.9	|101.28 +- 0.35s|
|toxic-xlm-roberta-base|92.56	|93.92	|86.71	|94.16	|94.21	|93.78	|80.63 +- 0.35s|
|dbmdz-bert-base-turkish-128k-uncased Embeddings + svc|90.9	|93.76	|85.1	|92.0	|91.31	|92.33	|10.35 +- 1.0s|
|tfidf Embeddings + lgbm|89.5	|89.16	|82.05	|90.96	|92.18	|93.14	|33.56 +- 0.37s|
|dbmdz-bert-base-turkish-128k-uncased Embeddings + catboost|88.37	|92.2	|81.86	|88.95	|88.3	|90.53	|14.75 +- 0.12s|
|tfidf Embeddings + xgb|87.61	|87.04	|78.67	|89.5	|90.63	|92.19	|55.15 +- 0.52s|
|dbmdz-bert-base-turkish-128k-uncased Embeddings + xgb|87.42	|91.74	|80.7	|87.7	|87.16	|89.81	|11.97 +- 0.17s|
|dbmdz-bert-base-turkish-128k-uncased Embeddings + lgbm|87.04	|91.28	|80.42	|86.59	|87.32	|89.6	|16.66 +- 0.15s|
|tfidf Embeddings + catboost|86.39	|85.45	|77.06	|88.36	|90.1	|90.98	|279.94 +- 2.28s|
|xlm-roberta-large Embeddings + lgbm|79.93	|88.31	|70.02	|79.7	|82.55	|79.06	|27.12 +- 0.66s|
|xlm-roberta-base Embeddings + catboost|79.05	|86.88	|70.3	|77.38	|83.23	|77.44	|16.6 +- 0.14s|
|xlm-roberta-large (Fine-Tuned) Embeddings + lgbm|78.95	|84.29	|72.2	|80.22	|80.29	|77.77	|186.16 +- 1.01s|
|fasttext Embeddings + catboost|78.4	|84.09	|65.69	|74.19	|84.98	|83.05	|4.35 +- 0.05s|
|xlm-roberta-base Embeddings + xgb|78.04	|86.58	|68.66	|75.58	|82.68	|76.67	|13.83 +- 0.14s|
|xlm-roberta-base Embeddings + lgbm|77.64	|86.41	|67.47	|75.69	|82.46	|76.15	|18.06 +- 0.08s|
|fasttext Embeddings + xgb|77.36	|83.19	|64.46	|71.92	|84.19	|83.05	|11.29 +- 0.13s|
|fasttext Embeddings + lgbm|76.92	|83.28	|63.65	|71.08	|84.18	|82.4	|1.97 +- 0.06s|
|toxic-xlm-roberta-large|72.01	|79.11	|73.34	|64.36	|65.67	|77.59	|151.79 +- 0.92s|
|xlm-roberta-base Embeddings + svc|58.73	|75.24	|49.77	|45.46	|62.15	|61.01	|11.87 +- 0.19s|
