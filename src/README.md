# Teknofest 2023 - Aşağılayıcı Söylemlerin Doğal Dil İşleme ile Tespiti
### Takım: L2 Regülasyon

---

### Çözüm Geliştirme Süreci Detayları

---

#### 1 - Eğitim Şeması / Parametreleri

##### 1.1 - StratifiedBatchSampler

Her iterasyon sırasında kullanılcak örneklem sayısı(batch_size) parametresi ve her örnekleme düşen veri dağılımı, eğitim performansı gelişimini hızlandırma veyahut stabilleştirmesine etki eder. Eğer düşük örneklem sayısı kullanılırsa, batch’lere düşen veri tek bir sınıftan oluşabilir, ya da tüm sınıfları içercek şekilde örneklem oluşturmaz. Bu da aşağıdaki sonuçlara yol açabilir.

  * Daha uzun eğitim süresi    
  * Sayıca daha az temsil edilen sınıflarda daha az tahmin performansı(daha az genelleşebilme)

Bunları engellemek için oluşturduğumuz çözümde, StratifiedBatchSampler sınıfı ile, eğitim sırasındaki örneklemlerin, her sınıftan eşit sayıda örnek barındırmasını sağladık.

Örneğin,  örneklem sayısı 8 ise, StratifiedBatchSampler kullanmadan,

	['OTHER', 'OTHER','OTHER','OTHER','OTHER','OTHER', 'SEXIST', 'RACIST']

şeklinde örneklem oluşabiliyorsa,    

StratifiedBatchSampler kullanırken, her örneklem aşağıdaki gibi her sınıftan yaklaşık olarak eşit sayıda örnek içerir.

	['PROFANITY', 'SEXIST', 'OTHER', 'SEXIST', 'RACIST', 'INSULT', 'OTHER', 'INSULT']

##### 1.2 - 'Early Stopping' Olmaksızın Eğitim

Eğitim sırasında test kümesine göre başarıyı takip ederek, belirli bir iterasyon boyunca başarı iyileşmiyorsa, overfittingi engellemek için kullanılan ‘early stopping’ tekniğini kullanmama kararı aldık. Çünkü, kullanılması halinde, eğitimi başarısını raporlayacağı kümedeki performans, maksimuma ulaştığında durduğu için, iyimser bir raporlama yapılmasına yol açmaktadır.


#### 2 - Model Validasyonu

##### 2.1 - Public - Private Folding

Projede denenen model mimarisi ve parametrelerinin başarı performansının, aynı validasyon yöntemi ve datayla raporlanması için örneklere fold atama sürecini utils/generate_data.py adlı bir scriptte gerçekleştirdik.

Teknofest tarafından verilen veriye, iki farklı seed ile, iki farklı fold tanımı yapıldı. Bu foldlara, public ve private isimleri verildi. Geliştirmeler ağırlık olarak public fold ile yapılırken, seyrek olarak da private fold ile public fold arasındaki korelasyona bakıldı. Bunun amacı, public foldda düzenli olarak iyileşme görürken, private’de aynı etkide gelişme görülmesini beklemekti. Aksi durum, skor gelişimi yapan geliştirmelerin, farklı seed’lere genelleşemediği, yani mevcut CV’ye overfit olabilme riski taşıdığını gösterecekti.

##### 2.2 - OOF Evaluation

Eğitilen modelin ne kadar başarılı olduğunun değerlendirmesini Out-Of-Fold skoru tekniğine göre yaptık. Bu teknik, veriyi bir cross-validation şemasına göre böldükten sonra, örneğin 5 Fold StratifiedKFold, her foldun eğitim kümesinde eğitim yapıp, test kümesini skorladıktan sonra, skorlanan test kümelerini birleştirir. Böylece, eğitim kümesindeki her örneğin, test setindeki performansına erişebildiğinizden, genelleşebilme performansını tam kapasiteyle test edebilmiş olmaktayız.

#### 3 - Model Başarı Takibi (Model Zoo)

Model geliştirme süreci boyunca yapılan hiperparametre ve model mimarisi seçimlerinden kaynaklanan performans değişimlerini takip etmek ve en iyilerini seçmek amacıyla, experiment tracking modülü geliştirdik. Bu yapıya src klasöründeki eğitim scriptleriyle yapılan denemeler –add-zoo parametresi eklenerek, deneylerin başarı performansları kayıt altına alınabilir.
	
Örneğin, aşağıdaki komut ile,

```python train_vector_stack.py -vector-model tfidf -head-model lgbm -experiment-name TFIDF_LGBM -fold-name public_fold --add-zoo```

eğitim tamamlandıktan sonra data/model_zoo.json dosyasına ‘TFIDF_LGBM’ adlı bir experiment kaydedilecektir.

---

#### 4 - Kodların Kullanımı

**train_bert.py**

Lorem ipsum
```
python train_bert.py --params
```

**train_embedding_stack.py**

Lorem ipsum
```
python train_embedding_stack.py --params
```

**train_vector_stack.py**

Lorem ipsum
```
python train_vector_stack.py --params
```
