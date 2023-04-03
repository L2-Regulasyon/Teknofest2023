# Teknofest 2023 - Aşağılayıcı Söylemlerin Doğal Dil İşleme ile Tespiti
### Takım: L2 Regülasyon

Bu kütüphane, L2 Regülasyon takımının `TeknoFest 2023` yarışması `Aşağılayıcı Söylemlerin Doğal Dil İşleme ile Tespiti` alt kolu için geliştirdiği mimarinin kaynak kodlarını içermektedir.

---

### Teknik Dökümanlar
Bu döküman sadece repository'nin kullanımına dair yönergeleri içermektedir. Aşağıdaki listeden ilgili bölümlere ait daha detaylı dökümanlara ulaşabilirsiniz.
- [Çözüm Geliştirme Süreci Detayları](src/README.md)
- [Denenen Modeller/Yaklaşımlar ve Sonuçları](src/models/README.md)
- [Nihai Çözüm Mimarisi ve Kullanılan Teknolojiler](SOLUTION.md)

---

## Kullanım Adımları

### Gereksinimler
- NVIDIA GPU
- Docker
- NVIDIA Container Toolkit

### 1. Docker gereksinimlerinin kurulması
- Docker: [Link](https://docs.docker.com/engine/install/)
- NVIDIA Container Toolkit: [Link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

### 2. Repository'in klonlanması
Bir klasörün içerisinde terminalinizi açıp aşağıdaki komutu çalıştırın.
```
git clone https://github.com/L2-Regulasyon/Teknofest2023.git
```
### 3. Docker ortamının oluşturulması
Projenin ana klasörüne gidin ve aşağıdaki komutu çalıştırın.
```
docker build -f Dockerfile -t l2reg .
```

**NOT:** Buradan itibaren olan adımlarda `$PWD` değişkeninin düzgün çalışabilmesi adına lütfen klasör yollarınızda boşluk karakteri veya kaçış karakteri barındırmamaya özen gösterin.
### 4. Server'ın çalıştırılması
Projenin ana klasörüne gidin ve aşağıdaki komutu çalıştırın.
```
docker run -v $PWD:/tmp/working \
-v ${HOME}/.cache:/container_cache \
-w=/tmp/working \
-e "XDG_CACHE_HOME=/container_cache" \
-p 7860:7860 --gpus all --rm -it l2reg \
python app.py
```

### 5. Eğitimlerin Tekrarlanması (Opsiyonel)
Bütün model setini tekrar eğitmek için aşağıdaki kodu çalıştırabilirsiniz.
```
docker run -v $PWD:/tmp/working \
-v ${HOME}/.cache:/container_cache \
-w=/tmp/working \
-e "XDG_CACHE_HOME=/container_cache" \
--gpus all --rm -it l2reg \
bash create_model_zoo.sh
```
Eğitim parametrelerini değiştirerek daha özelleştirilmiş eğitimler koşmak istiyorsanız [bu dökümanı](src/README.md) inceleyebilirsiniz.


### 6. Analizlerin Tekrarlanması (Opsiyonel)
Jupyter sunucusunu aşağıdaki kod ile başlatarak `analysis` altındaki notebook dosyalarını yeniden çalıştırabilirsiniz.
```
docker run -v $PWD:/tmp/working \
-v ${HOME}/.cache:/container_cache \
-w=/tmp/working \
-e "XDG_CACHE_HOME=/container_cache" \
-p 8888:8888 --ipc=host \
--gpus all --rm -it l2reg \
jupyter notebook --no-browser --ip="0.0.0.0" \
--notebook-dir=/tmp/working --allow-root
```