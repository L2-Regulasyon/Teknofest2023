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

### 5. Eğitim ve analiz kodlarının çalıştırılması (Opsiyonel)
İlgili süreçlere [buradan](src/README.md) ulaşabilirsiniz.
