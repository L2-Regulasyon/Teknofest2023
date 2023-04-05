# Teknofest 2023 - Aşağılayıcı Söylemlerin Doğal Dil İşleme ile Tespiti
### Takım: L2 Regülasyon

---

### Çözüm Geliştirme Süreci Detayları

---

## 1 - Eğitim Şeması / Parametreleri

### 1.1. StratifiedBatchSampler - [Referans](https://discuss.pytorch.org/t/how-to-enable-the-dataloader-to-sample-from-each-class-with-equal-probability/911/7)

Her iterasyon sırasında kullanılacak `batch_size` parametresi ve her örnekleme düşen veri dağılımı, eğitim performansı gelişimini hızlandırma veyahut stabilleştirmesine etki eder. Eğer düşük örneklem sayısı kullanılırsa, batch’lere düşen veri tek bir sınıftan oluşabilir, ya da tüm sınıfları içercek şekilde örneklem oluşturmaz. Bu da aşağıdaki sonuçlara yol açabilir.

  * Daha uzun eğitim süresi    
  * Sayıca daha az temsil edilen sınıflarda daha az tahmin performansı (daha az genelleşebilme)

Bunları engellemek için oluşturduğumuz çözümde `StratifiedBatchSampler` sınıfı ile eğitim sırasındaki mini-batchlerin her sınıftan eşit sayıda örnek barındırmasını sağladık.

Örneğin,  örneklem sayısı 8 ise, StratifiedBatchSampler kullanmadan:

	['OTHER', 'OTHER','OTHER','OTHER','OTHER','OTHER', 'SEXIST', 'RACIST']

şeklinde örneklem oluşurken; StratifiedBatchSampler kullanarak her sınıftan aşağıdaki şekilde yaklaşık olarak eşit sayıda örnek toplanabilir:

	['PROFANITY', 'SEXIST', 'OTHER', 'SEXIST', 'RACIST', 'INSULT', 'OTHER', 'INSULT']

Bu teknik eğitim-validasyon skorumuz arasındaki açıklığı azaltmada etkili olmuştur.

### 1.2. 'Early Stopping' Kapalı Eğitim

Eğitim sırasında test kümesine göre başarıyı takip ederek belirli bir iterasyon boyunca başarı iyileşmiyorsa overfittingi engellemek için kullanılan ‘early stopping’ tekniğini kullanmama kararı aldık. Bu metot kullanıldığında raporlayacağı kümedeki performans azalmaya başladığında süreci durdurduğu için, iyimser bir raporlama yapılmasına yol açacaktır. Eğitim sürecimizi gerçek hayat senaryolarındaki gibi _test verisini bilmeyeceğimizi_ varsayarak tasarladığımız için cross-validation süreci boyunca da test verisinden alınan herhangi bir bilginin eğitim sürecini etkilemesine izin vermedik.

### 1.3. Online Hard Example Mining (OHEM) - [Referans](https://arxiv.org/abs/1604.03540v1)

Modeli multi-class problem için eğitebilmek adına Cross-Entropy loss fonksiyonunu `Online Hard Example Mining` yaklaşımı ile kullandık. Bu yaklaşım, kolay örneklerin loss'u domine edip zor örneklerinin öneminin azaldığı durumları engellemek için kullanılıyor. Model'in örneklerden aldığı loss'un sadece en yüksek `%k` lık bir dilimi hesaba katılıyor. Böylelikle model ne kadar iyileşirse iyileşsin hep örneklerin en zor `%k` lık diliminden loss feedback alıyor.

### 1.4. Sınıf Ağırlıklandırma

Her sınıfın eğitim örneği eşit sayıda olmadığı için az temsil edilen sınıfların da öğrenimini iyileştirmek adına sınıf ağırlıkları _(az temsil edilen sınıf, daha önemli olacak şekilde)_ belirledik.

Ağırlıklar, aşağıdaki yaklaşım ile belirlendi. Örnekte her sınıfın diğer sınıflara göre bulunma katsayısının **küpkökü** oranında değer düşüşü yaşamaktadır.
```
cls_weights = list(dict(sorted(dict(1 / ((y_train.value_counts(normalize=True)) ** (1 / 3))).items())).values())
cls_weights /= min(cls_weights)
```

### 1.5 Cosine Scheduler + Warm Up - [Referans](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)

Belirli bir ısınma süreci boyunca hedef learning rate’a kadar küçük oranlarla artan, belirlenen learning rate’e ulaştığında eğitim aşaması uzadıkça learning rate’i düşürecek Cosine Scheduler tekniğini kullandık. Bu teknik özellikle fine-tuning eğitimlerinde halihazırdaki weight'leri daha ilk iterasyonlarda aşırı değiştirip modelin bütün embedding yapısını bozmamak adına önemlidir.

### 1.6. Gradient Clipping - [Referans](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem#:~:text=What%20is%20gradient%20clipping%3F,gradients%20to%20update%20the%20weights.)


Eğitilen parametrelerin büyüklüklerini belirli bir değeri geçmeyecek şekilde sınırlayan Gradient Clipping tekniğini kullandık. Bu teknik, tahminleri belirli parametrelerin domine etmesindense gradyanların genele yayılıp parametreler üstünde regülarizasyon etkisi yaratılmasını sağlıyor.

### 1.7. Weight Decay - [Referans](https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab)

Weight Decay, loss fonksiyonuna overfit’i engellemek için parametre büyüklüğüne göre penaltı ekler. Overfitting’i azaltmak için bu metodu da parametrik olarak mevcut akışımıza ekledik. Bu penaltıya göre, sinir ağı eğitilirken, mevcut iterasyonda büyük parametreler kullanılıyorsa, daha büyük loss elde edilirken, örneğin L2 norm uygulandığı durumda elde edilen küçük parametreler ile daha küçük loss elde edilir. Böylelikle, sinir ağının kararına parametrelerin büyüklük olarak önemli bir alt kümesinden ziyade, parametrelerin geneli karar verdiğinden, modelin farklı veri desenlerine genelleşmesi daha olası hale gelmektedir.

### 1.8. Label Smoothing - [Referans](https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06)

Label Smoothing, loss fonksiyonu olarak, cross-entropy kullanırken, sinir ağı mimarisinin eğitim verisine overfit olmasını engelleyen diğer bir regülarizasyon tekniği kullandık.. Bu teknik, modelin doğru sınıf üzerindeki kararlılığını azaltarak, görülmeyen verinin eğitim verisine benzeme varsayımı konusunda daha az katı modeller eğitmeye yaramaktadır.

### 1.9. LLRD Decay - [Referans](https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e)

Model mimarilerinin embedding ve encoder katmanlarına regülarizasyonu arttıracak parametreler ekleyerek, overfit’i azaltmak istedik.

### 1.10. Masked Language Modeling - [Referans](https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling)

Fine-tune ettiğimiz modelin kullanacağı kelimelerin anlam temsillerini iyileştirmek ve veri bağlamını daha iyi anlatabilmek adına; metindeki bazı kelimeleri gizleyip gizlenmiş kısmı modele tahmin ettirdiğimiz bir eğitim tekniği kullanmayı denedik.  Modellerin alan spesifik bir bağlamı öğrenmeleri için, kelimelerin anlamlarının ve kelimelerin bir araya gelmesinden oluşan anlamların öğretilebilmesi gerekmektedir. Bu eğitimler, bir metnin içerisindeki bazı kelimeler gizlenip/değiştirilip, bağlama uyan doğru kelimeyi bulabilme ödülü ile eğitilir. Bu yaklaşım dil modeli eğitilirken kullanıldığı gibi, fine-tune ederken de kullanılabilir. Böylece mevcut modele yeni veri setindeki bağlama uyum sağlatarak modelin bir ısınma (pre-training) sürecinden geçmiş olması sağlanır.

### 1.11. Model-Data Unbiasing

Yarışmada kullandığımız veriyi inceleyerek, bu verinin gerçek dünya verisinde _(mevcut eğitim setinden daha büyük ve çeşitli örneklemde)_ hangi zorluklarla karşılaşabilceğini tespit ettik. Belirlediğimiz aksiyonlarla da final çözümümüzde modelimizin yarışma verisine karşı olan önyargılarına karşı önlemler almaya çalıştık.

Tespit ettiğimiz model önyargıları;

- **Cümle uzunlukları:** Yukarıdaki temel analiz çıktısında da bahsedildiği üzere `OTHER` sınıfına ait cümleler diğerlerine göre belirgin derecede daha uzun. Bu da modelin cümleler uzadıkça tahminini `OTHER` sınıfına kaydırmasına neden oluyor.
- **Büyük harf dağılım dengesizliği:** Büyük harf içeren kelime kullanımının `OTHER` sınıfında neredeyse hiç yokken diğer sınıflarda `%30` civarında olduğunu görüyoruz. Böylelikle model yarattığımız işaretçiye gereğinden fazla anlam yükleyebiliyor. Masum bir kelimenin baş harfini büyütünde model ofansif sınıflar ile etiketlemeye meylediyor. `Uncased` yerine `Cased` model kullanılan herhangi bir senaryoda model bunu istemsizce özel işaretçiye gerek duymadan _kendisi yapıyor_.
- **Cinsiyetçi Önyargı:** Modelde cinsiyetlere ait kelimeler kullanıldığında sınıflandırmalar cümle uzamadığı sürece ofansife kayıyor.
- **Hitabet eksikliği:** `OTHER` sınıfına ait çoğu cümle ya üçüncü kişiye yönelik, ya da tanım-açıklama formatında yazılmış. Ofansif kategoriye girecek cümleler ise çoğunlukla ikili konuşmalardan alınan örnekler. Bu yüzden model genelgeçer ikili muhabbete ait jargon-kelime gördüğünde sınıflandırmasını belirgin bir şekilde ofansife kaydırıyor.

### 1.12. Voting Ensemble

Lorem Ipsum

## 2. Model Validasyonu

### 2.1. Public - Private Folds

Projede denenen model mimarilerinin başarı performansının, aynı validasyon yöntemi ve test verisi ile raporlanması için örneklere fold atama süreci [generate_data.py](generate_data.py) adlı scriptte gerçekleştirildi.

Organizatör tarafından verilen veriye iki farklı seed ile, iki farklı fold tanımı yapıldı. Bu foldlara `public` ve `private` isimleri verildi. Geliştirmeler ağırlık olarak `public` fold ile yapılırken, seyrek olarak da `private` fold ile `public` fold arasındaki korelasyona bakıldı. Bunun amacı, public fold'da düzenli olarak iyileşme görürken private’de aynı etkide gelişme görülmesini beklemekti. Aksi durum; skor gelişimi yapan geliştirmelerin farklı seed’lere genelleşemediği, yani mevcut CV’ye overfit olabilme riski olduğunu gösterecekti.

### 2.2. Out-of-Fold (OOF) Evaluation - [Referans](https://machinelearningmastery.com/out-of-fold-predictions-in-machine-learning/#:%7E:text=An%20out%2Dof%2Dfold%20prediction,example%20in%20the%20training%20dataset.)


Eğitilen modelin ne kadar başarılı olduğunu Out-of-Fold tekniğiniyle skorlayarak değerlendirdik. Bu teknik, veriyi bir cross-validation şemasına göre böldükten sonra _(örneğin 5 Fold StratifiedKFold)_ her foldun eğitim kümesinde eğitim yapıp test kümesini skorladıktan sonra, skorlanan test kümelerini birleştirir. Böylece, eğitim kümesindeki her örneğin test setindeki performansına erişilebilir ve bütün veriye ait tek bir genelleşebilme performansı metriği üretilebilir.

## 3. Model Başarı Takibi (Model Zoo)

Model geliştirme süreci boyunca yapılan hiperparametre ve model mimarisi seçimlerinden kaynaklanan performans değişimlerini takip etmek ve en iyilerini seçmek amacıyla bir deney takip modülü geliştirdik. `src` klasöründeki eğitim kodlarına `–-add-zoo` parametresi eklenerek deneylerin başarı performansları kayıt altına alınabilir.
	
Örneğin, aşağıdaki komut ile eğitim tamamlandıktan sonra `data/model_zoo.json` dosyasına `TFIDF_LGBM` adlı bir deney sonucu kaydedilecektir.

```
python train_vector_stack.py -vector-model tfidf -head-model lgbm -experiment-name TFIDF_LGBM --add-zoo
```

---

## 4. Kodların Kullanımı

Buradaki kodları kullanmadan önce ana dizinden bu kodların olduğu dizine aşağıdaki kod ile geçiş yapmalısınız:
```
cd src
```


**generate_data.py**

Hiçbir parametre ayarlamadan sonuçlarımızı yeniden alabileceğiniz işlenmiş veriyi aşağıdaki kodu çalıştırarak elde edebilirsiniz. Dilerseniz üretilmesini istediğiniz cross-validation fold sayısını kodun içerisinden ayarlayabilirsiniz.
```
python generate_data.py
```

**train_bert.py**

`BERT` ve `RoBERTa` mimarili modelleri eğitmek için bu trainer'ı kullanabilirsiniz. Aşağıdaki parametreler ile özelleştirilebilir:
- **-model-path:** Kullanılacak baz modelin adresi
- **-epochs:** Modelin kaç epoch eğitileceği
- **-batch-size:** Batch size
- **-tokenizer-max-len:** Tokenizer'ın cümleleri maksimum kaç token'a kadar işleyeceği
- **-learning-rate:** Learning rate
- **-warmup-ratio:** Cosine LR Scheduler'ın warm-up süresinin iterasyon yüzdesi cinsinden karşılığı
- **-weight-decay:** Weightlerin L2 normuna uygulanacak decay-rate
- **-llrd-decay:** Model içi layerlar arası LR decay-rate
- **-label-smoothing:** Model sınıf değerlerine uygulanacak yumuşatma oranı
- **-grad-clip:** Backpropagation esnasında gradyanların kırpılacağı eşik değeri
- **-prevent-bias:** Verideki ön-yargıya karşı alınacak önlem seviyesi _(0,1,2)_
- **--mlm-pretrain:** Baz modelde unsupervised Masked-Language-Modelling ön eğitimi uygulaması _(Açık-Kapalı)_
- **-mlm-probability:** MLM eğitimi esnasında tokenların maskelenme olasılığı
- **--cv:** Eğitimde cross-validation kullanılması (Açık-Kapalı)
- **-out-folder:** Model sonuçlarının çıkartılacağı dosya
```
python train_bert.py **params
```

**train_embedding_stack.py**

`BERT` ve `RoBERTa` mimarili modelleri `SVC` ve `GBDT` tipi ağaç modellerini sınıflandırma katmanı olarak kullanarak eğitmek için bu trainer'ı kullanabilirsiniz. Aşağıdaki parametreler ile özelleştirilebilir:
- **-embedding-model-path:** Kullanılacak embedding baz modelin adresi
- **-head-model:** Sınıflandırma için kullanılacak modelin tipi _(SVC, LGBM, CatBoost, XGBoost)_
- **--retrain-embed-model:** Baz modelin tekrar eğitilmesi _(Açık-Kapalı)_

```
python train_embedding_stack.py **params
```

**train_vector_stack.py**

`TFIDF` ve `FastText` modellerini `GBDT` tipi ağaç modellerini sınıflandırma katmanı olarak kullanarak eğitmek için bu trainer'ı kullanabilirsiniz. Aşağıdaki parametreler ile özelleştirilebilir:

- **-vector-model:** Kullanılacak embedding modeli _(TFIDF, FastText)_
- **-head-model:** Sınıflandırma için kullanılacak modelin tipi _(LGBM, CatBoost, XGBoost)_

```
python train_vector_stack.py **params
```

**create_voting_bert.py**

`train_bert.py` ile CV modunda eğitilmiş modellerin bnirleştirilip tek bir voting-ensemble haline getirilmesini kodu aşağıdaki parametrelerle özelleştirerek sağlayabilirsiniz:
- **-c:** Kullanılacak modellerin adresleri
- **-o:** Oluşturulacak ensemble model'in kaydedilme adresi

```
python create_voting_bert.py **params
```

**push_to_hub.py**

`train_bert.py` ile CV kullanılmadan bütün veri ile eğitilmiş modelleri HuggingFace'a bu kod ile yükleyebilirsiniz. Yükleyeceğiniz modeli ve hedef repository'i koda müdahale ederek değiştirmeniz gerekmektedir.
```
python push_to_hub.py
```
