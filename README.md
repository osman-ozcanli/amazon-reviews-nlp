# Amazon Reviews NLP - Sentiment Analysis

Amazon elektronik urun yorumlari uzerinde TF-IDF tabanli duygu analizi (sentiment analysis).
Klasik, yorumlanabilir bir lineer model hatti; cross-validation, confounding kontrolu ve
problem formulasyonu ile metodolojik olarak saglamlastirilmistir.

## Sonuclar (Results)

| Kurgu                     | macro-F1 | Accuracy |
|---------------------------|----------|----------|
| 3-class (neg / neu / pos) | 0.66     | 0.82     |
| **2-class (neg / pos)**   | **0.86** | **0.93** |

- Notr (3-yildiz) sinif dil olarak belirsiz oldugundan cokuyor; notr cikarilinca macro-F1 +0.20.
- Model: LogisticRegression (macro-F1'e gore reproducible secim, StratifiedKFold CV ile dogrulandi).
- NER bulgusu text-length confounding kontrolu ile duzeltildi (entity-yogun degil, daha UZUN yorumlar).

## Veri (Data)

> Veri ve egitilmis modeller boyut nedeniyle repoya DAHIL DEGILDIR (`.gitignore` ile haric).

- **Orijinal:** ~1.69M Amazon elektronik urun yorumu.
- **Kullanilan:** Orijinalden alinmis 400k'lik stratified sample.
- **Beklenen dosya:** `amazon_reviews_400k.csv`
- **Notebook'taki yol (H5):**
  `/content/drive/MyDrive/NLP Project-Colab Only/amazon_reviews_400k.csv`

**Calistirmak icin:** Veriyi kendi Google Drive'iniza yukleyin ve notebook'taki
`pd.read_csv(...)` hucresindeki (H5) yolu kendi dosya yolunuza gore guncelleyin. Benim yolum asagidaki gibidir.
`df=pd.read_csv('/content/drive/MyDrive/NLP Project-Colab Only/amazon_reviews_400k.csv')`

Notebook Google Colab icin tasarlanmistir (`drive.mount`).

## Dosyalar (Files)

- `MY_1_Amazon_Reviews_NLP_04_06_2026_vers_final.ipynb` — final notebook (Colab).
- `_local_validation/` — adim adim yerel (20k) dogrulama scriptleri.

## Teknolojiler

Python, scikit-learn (TF-IDF, LogisticRegression/SGD/LinearSVC), spaCy (NER),
pandas, matplotlib/seaborn, wordcloud.