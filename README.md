# WhatsApp Sohbet Analizi

`_chat.txt` dosyasını okuyarak kişi başı istatistikler üretir, PNG grafikler ve tek bir PDF rapor çıktısı oluşturur.

## Özellikler

- Kişi adlarını dosyadan otomatik algılar (elle giriş gerekmez)
- Mesaj sayısı, kelime/karakter sayısı, medya paylaşımı
- Yanıt süreleri (ortalama, medyan, maksimum, geç yanıt sayısı)
- Konuşma başlatma analizi
- Saatlik ve günlük aktivite dağılımı
- Aylık trend grafiği
- Tüm grafikleri birleştiren tek PDF rapor

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```bash
# Varsayılan: mevcut dizindeki _chat.txt → chat_analysis/ klasörüne çıktı
python app.py

# Özel dosya ve çıktı klasörü
python app.py /path/to/_chat.txt /path/to/output_dir
```

Çıktı klasöründe PNG grafikler ve `rapor.pdf` oluşturulur.

## Gereksinimler

- Python 3.10+
- Bağımlılıklar: `requirements.txt`

## Desteklenen Format

WhatsApp'ın dışa aktardığı standart `.txt` formatı:

```
[G.AA.YYYY, SS:DD:SS] Kişi Adı: mesaj içeriği
```
