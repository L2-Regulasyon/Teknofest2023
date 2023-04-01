# Teknofest 2023 - Aşağılayıcı Söylemlerin Doğal Dil İşleme ile Tespiti
### Takım: L2 Regülasyon

---

### Literatür Taraması

Aşağılayıcı söylemleri sınıflandırma görevi için literatürde farklı yaklaşımlar öneren birçok yayın bulunmaktadır. Yarışma süresince yararlandığımız kaynaklar listelenmiştir.   

[Aken B. V., Risch J. Krestel R. ve Löser A., 2018](https://arxiv.org/abs/1809.07572) `Other, Toxic, Obscene, Insult, Identity Hate, Severe Toxic ve Threat` sınıflarını tahmin etmek için yaptıkları çalışmada en iyi skoru`(F1:0.783)` ile `Bidirectional GRU Attention (FastText)` modeli ile elde etmişlerdir. Ancak kullandıkları diğer modellerde de skorların benzer olduğu gözlemlenmiştir.

[Duchêne C., Jamet H., Guillaume P. ve Dehak R., 2023](https://arxiv.org/abs/2301.11125) `Toxicity, Obscene, Sexual Explict, Identity Attack, Insult ve Threat` sınıflarının tahmini için `BERT, RNN, XLNET` altyapılarını kullanan modeller ile denemeler yapmışlardır ve bütün bu modellerin benzer sonuçlar verdiğini vurgulamışlardır. Nihai olarak; `Focall Loss` ile eğitilmiş `RoBERTa` modeli AUROC ve F1 olarak en iyi sonucu veren model olmuştur. 

### Denenen Dış Veriler


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
