# PROGRESS — MY-1 Amazon Reviews NLP İyileştirme

Bu dosya, **Dosya 18** (`0.BU KLASOR VE PROJEYI ANALIZ ETTIRDIM 18.txt`) analiz raporundaki
düzeltme planının adım adım uygulanmasını kaydeder.

- **Hedef notebook / target:** `MY-1-Amazon Reviews NLP-01.06.2026.ipynb` (TF-IDF + linear, 3 sınıf)
- **Çalışma düzeni / workflow:** Hibrit — kod yerelde `sample_20k.csv` ile doğrulanır,
  ardından full 400k Colab'da çalıştırılır.
- **Kural:** Bir adım yalnızca kullanıcı Colab'da çalıştırıp **onayladıktan** sonra
  "onaylandı" olarak işaretlenir.

---

## Plan (5 Adım)

| Adım | Konu | Durum |
|------|------|-------|
| 1 | Veri-seviyesi fix: `dropna(subset=["reviewText"])` + `df_analysis.copy()` | ✅ Onaylandı |
| 2 | Model fix: `clone` tfidf + isimle best-model seçimi + cross-validation | ✅ Onaylandı |
| 3 | Nötr sınıf krizi analizi + 2-class (pos/neg) deneyi | ✅ Onaylandı |
| 4 | NER confounding: text-length normalizasyonu | ✅ Onaylandı |
| 5 | Çok dilli veri notu + kod kalitesi düzeltmeleri | ✅ Onaylandı |
| 5b | Konumlandırma: Adım 5 parçalarını doğru bölümlere taşı (final akış) | ✅ Onaylandı |

---

## Kayıtlar

<!-- Her adim onaylandikca buraya islenecek -->

### ✅ Adım 1 — Veri-seviyesi fix (Onaylandı)

**Sorun (Dosya 18, Problem 3 + Bug 1):**
- `df.dropna(axis=0)` herhangi bir kolonda null olan TÜM satırları siliyordu → çoğu
  `reviewerName` null'u olan **geçerli yorumlar** boşuna siliniyordu.
- `df_analysis = df[[...]]` bir slice'tı; `.copy()` olmadığı için `df_analysis['sentiment']`
  ataması `SettingWithCopyWarning` veriyordu (tanımsız davranış).

**Yapılan değişiklik:**
- Hücre 12: `df.dropna(axis=0, inplace=True)` → `df.dropna(subset=["reviewText"], inplace=True)`
- Hücre 27: `df[[...]]` → `df[[...]].copy()`

**Yerel doğrulama (sample_20k.csv):**
- Eski yöntem: 323 satır siliniyordu | Yeni yöntem: yalnızca 8 satır → **315 geçerli satır kurtarıldı**
  (oran 400k'da ~5997 satıra denk gelir).
- `.copy()` sonrası `SettingWithCopyWarning` tamamen kayboldu (warning'i hata olarak test ettim).
- Sentiment dağılımı bozulmadı.
- Doğrulama script'i: `_local_validation/step1_data_fix.py`

### ✅ Adım 2 — Model fix (Onaylandı)

**Sorun (Dosya 18, Bug 2 + Bug 3 + Problem 2):**
- Bug 2: tek `tfidf` nesnesi 3 pipeline'da paylaşılıyordu (kırılgan, mutable state).
- Bug 3: `summary_df.iloc[1]` ile best-model **pozisyona** göre (kırılgan) seçiliyordu.
- Problem 2: tek train/test split → genelleme performansı doğrulanmamıştı.

**Yapılan değişiklik:**
- `("tfidf", tfidf)` → `("tfidf", clone(tfidf))` (her pipeline bağımsız vectorizer).
- Best-model seçimi: `summary_df.loc[summary_df["CV Macro F1"].idxmax(), "Model"]` (SEÇENEK A — Logistic).
- `StratifiedKFold(n_splits=2)` + `cross_val_score(scoring="f1_macro")` eklendi.

**Colab 400k sonuçları:**
| Model | CV Macro F1 | Test Macro F1 | Accuracy |
|---|---|---|---|
| **Logistic** | **0.6520** | 0.6616 | 0.8213 |
| LinearSVC | 0.6469 | 0.6587 | 0.8638 |
| SGD | 0.5857 | 0.5813 | 0.8645 |

- **Kritik bulgu:** CV ≈ Test (fark ≤0.012) → **overfitting yok**, model genelliyor.
- Seçilen model: **Logistic** (SEÇENEK A, macro-F1'e göre reproducible).
- Eski tek-split skorları güvenilirmiş; artık bunu CV ile *biliyoruz*.

### ✅ Adım 3 — Nötr sınıf krizi + 2-class deneyi (Onaylandı) ⭐ EN ÖNEMLİ DEĞİŞİKLİK

**Sorun (Dosya 18, Problem 1):**
3-class modelde nötr sınıf (3-yıldız) çöküyordu — recall ~0.07–0.35 arası, f1 ~0.31.
Sebep: 3-yıldız yorumlar dil olarak pozitif/negatif arası **belirsiz** (düşük duygu sinyali /
low sentiment signal). Error analysis de bunu doğruluyordu: en sık karışma **neu↔pos**
(1→2: 2999, 2→1: 2220 yanlış). Orijinal projede bu kriz hiç analiz edilmemişti.

**Yapılan değişiklik:**
- 3-class modelin sınıf-bazlı metrikleri açıkça raporlandı (nötr çöküşü belgelendi).
- **2-class fallback deneyi eklendi:** nötr (sentiment==1) çıkarıldı, problem binary
  (neg=0 vs pos=1) hale getirildi. Yeni hücreler error analysis'in **hemen ardına** kondu
  (anlatı bütünlüğü: "nötr karışıyor → çıkaralım").
- Değişken isimleri ayrı (`df_bin`, `xb_*`, `yb_*`, `bin_pipe`) → 3-class hikâyesi (error_df,
  NER) bozulmadan korundu. Hiçbir mevcut hücre silinmedi.

**Sonuç — 3-class vs 2-class (400k, Logistic):**
| Kurgu | macro-F1 | Accuracy |
|---|---|---|
| 3-class | 0.66 | 0.82 |
| **2-class** | **0.86** | **0.93** |

Nötrü çıkarmak → macro-F1 **+0.20**, accuracy **+0.11**. Dosya 18'in tezi 400k'da tam doğrulandı.

**2-class sınıf bazlı (400k):**
| Sınıf | precision | recall | f1 | support |
|---|---|---|---|---|
| neg | 0.66 | 0.90 | 0.76 | 9059 |
| pos | 0.99 | 0.93 | 0.96 | 64217 |
- accuracy 0.9306 | macro-F1 0.8613

**Yerel (20k) vs Colab (400k) — ölçek etkisi:**
| Metrik | 20k | 400k | Δ |
|---|---|---|---|
| neg recall | 0.78 | 0.90 | +0.12 |
| neg f1 | 0.69 | 0.76 | +0.07 |
| macro-F1 | 0.817 | 0.861 | +0.044 |
| accuracy | 0.913 | 0.931 | +0.018 |
- **Çıkarım:** İyileşmenin tamamı **azınlık (neg) sınıfında**. Daha çok veri en çok
  az-temsil edilen sınıfa yarar (20k'da neg ~447 test örneği, 400k'da ~9000). Pos zaten doymuştu.
- Doğrulama script'i: `_local_validation/step3_neutral_binary.py`

### ✅ Adım 4 — NER confounding kontrolü (Onaylandı) ⭐ BULGU DEĞİŞTİ

**Sorun (Dosya 18, Problem 5):**
Orijinal NER analizi "yanlış sınıflananlar daha çok entity içeriyor" (ort. 2.00 vs 1.43) sonucunu
**text-length kontrol etmeden** çıkarmıştı. Uzun metin → hem daha çok entity hem daha zor
sınıflandırma; yani fark sadece uzunluk confound'u olabilirdi.

**Yapılan değişiklik:**
- `word_count` ve `entity_density` (entity/kelime) eklendi; ham sayı yerine **yoğunluk** karşılaştırıldı.
- `scipy.stats.mannwhitneyu` ile istatistik testi eklendi.

**Colab 400k sonucu (spaCy, n=10000+10000):**
| Ölçüm | correct | error | oran | p |
|---|---|---|---|---|
| word_count (uzunluk) | 105.5 | 148.1 | 1.404 | 7.4e-129 |
| entity_count (ham) | 1.43 | 2.00 | 1.403 | 8.8e-44 |
| entity_density (norm.) | 0.0129 | 0.0131 | **1.011** | 3.3e-06 |

**Kritik çıkarım — bulgu DEĞİŞTİ:**
- Ham entity oranı (1.403) ≈ kelime sayısı oranı (1.404) → "daha çok entity" bulgusu neredeyse
  **tamamen uzunluk confound'u**. Normalize edince fark %40 → %1'e çöktü.
- p=3.3e-06 yanıltıcı: n=10000 olduğu için %1 bile "anlamlı" çıkar; **pratik etki yok**
  (istatistiksel anlamlılık ≠ pratik anlamlılık).
- **Düzeltilmiş bulgu:** Modelin zorlandığı yorumlar "entity-yoğun" değil, basitçe **DAHA UZUN**
  yorumlar. Uzun yorumlar daha karışık/dengeli görüş + nüans içerdiğinden TF-IDF için daha zor.
- Hücre 12 (CONCLUSION FOR NER) bu doğrultuda **güncellendi** (dürüst, daha doğru bir sonuç).
- Doğrulama script'i (kod yolu, spaCy lokalde yok): `_local_validation/step4_ner_confounding.py`

### ✅ Adım 5 — Çok dilli veri notu + kod kalitesi (Onaylandı)

> ✅ **Akış notu (ÇÖZÜLDÜ — bkz. Adım 5b):** Parçalar başta kod bloğunun sonuna (polish bloğu)
> eklenmişti; Adım 5b'de doğru bölümlerine taşındı. Bu not tarihsel kayıt olarak bırakılmıştır.

**Sorun (Dosya 18, Problem 4 + Kod Kalitesi):**
- Problem 4: çok dilli (İspanyolca/Portekizce) yorumlar mention edilmemişti (sessiz gürültü).
- `def wc(...)` içinde `wc=WordCloud(...)` fonksiyon adıyla çakışıyordu (kötü pratik).
- `en_core_web_sm` küçük model; NER doğruluğu sınırlı — not edilmemişti.

**Yapılan değişiklik (3 noktasal rötuş):**
- **Parça 1 — Çok dilli not:** dependency-free, vektörize heuristic (yeterince uzun ama hiç yaygın
  İngilizce stopword içermeyen yorum → muhtemelen non-English) + markdown not. Ön-işleme/EDA
  bölümüne ait.
- **Parça 2 — `wc` fix:** iç değişken `wc` → `cloud` (fonksiyon adıyla çakışma giderildi),
  docstring eklendi. WordCloud bölümüne ait.
- **Parça 3 — spaCy notu:** `en_core_web_sm`'in sınırlı NER doğruluğu markdown ile not edildi.
  NER bölümüne ait.

**Yerel doğrulama (sample_20k.csv):**
- Çok dilli: 22/19992 (~%0.11) muhtemelen non-English → 400k'da kabaca ~440 yorum. Örnekler net
  İspanyolca/Portekizce; oran çok düşük olduğu için **filtrelenmedi, sadece not edildi**.
- Doğrulama script'i: `_local_validation/step5_multilingual.py`

**Kalan adımlar (tamamlandı — Adım 5b'de doğrulandı):**
- ✅ CONCLUSION FOR NER → Adım 4 doğrultusunda güncellenmiş metin.
- ✅ Ana CONCLUSION → tüm değişikliklerin (3-class→2-class, CV, confounding) özeti.
- ✅ Model kaydı: hem `sentiment_model_tfidf.joblib` (3-class) hem `..._binary.joblib` (2-class).

### ✅ Adım 5b — Konumlandırma / final akış (Onaylandı) ⭐ AKIŞ NOTU ÇÖZÜLDÜ

**Sorun:** Adım 5'in 3 parçası + 2 başlık/kapanış markdown'ı kod bloğunun sonunda "polish bloğu"
olarak duruyordu (eski nb hücre 89–96); anlatı akışı bölük, Restart & Run All sırası mantıksızdı.

**Yapılan değişiklik (yeni dosya: `MY_1_Amazon_Reviews_NLP_04_06_2026_vers_final.ipynb`):**
- **Parça 1** (çok dilli not md + tespit kodu) → temizlik (h29) sonrası, "Labeling" öncesine taşındı.
- **Parça 2** (`wc` fix) → WordCloud bölümündeki eski `def wc` hücresi bununla **değiştirildi**
  (tek `def wc`, iç değişken `cloud`; eski `wc=WordCloud` kalıntısı yok).
- **Parça 3** (spaCy notu) → NER bölümünde `spacy.load` hücresinin **üstüne** markdown olarak kondu.
- Polish bloğu (başlık + boş + kapanış md'leri dahil) **tamamen kaldırıldı**. Hücre: 103 → 98.

**Yerel doğrulama (sample_20k.csv):** `_local_validation/step5b_placement.py` — 3 parça da yeni
konumlarında, o noktada tanımlı değişkenlerle (temiz `df_model`, `WordCloud`/`STOPWORDS`)
kırılmadan çalıştı (izole isim-çakışması testi dahil).

**Colab 400k teyidi (executed dosya incelendi):**
- 59/59 kod hücresi çalışmış, **0 hata**. Akış yukarıdan aşağı monoton, `df_analysis` tanım→kullanım doğru.
- Parça 1 çıktı: **501 non-EN yorum (%0.13)**. Model comparison Logistic CV 0.652 / Test 0.662
  (önceki 400k ile birebir). 2-class macro-F1 **0.8613**, acc **0.9306**. Modeller kaydedildi.
- ⚠️ Kozmetik: bir **boş kod hücresi** (h52, orijinalden) `execution_count`'ta tekrar yaratıyor;
  fonksiyonel etki yok. İsteğe bağlı: boş hücreyi sil + Restart & Run All → sayılar 1..58 ardışık olur.

> **PROJE DURUMU: tüm adımlar (1–5b) tamamlandı ve Colab 400k'da doğrulandı.** Sıradaki iş: git/GitHub.
