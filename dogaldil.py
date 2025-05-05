import os
import re
import pandas as pd
from gensim.models import Word2Vec
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# NLTK verilerini indir
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Klasör yapısını oluştur
os.makedirs("word2vec_models/lemmatized", exist_ok=True)
os.makedirs("word2vec_models/stemmed", exist_ok=True)
os.makedirs("zipf_analizi", exist_ok=True)
os.makedirs("temizlenmis_veriler", exist_ok=True)
os.makedirs("processed_data", exist_ok=True)

# NLP araçları
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Veri dosyaları
txt_files = [
    "gazeteler/20250403.txt",
    "gazeteler/20250404.txt",
    "gazeteler/20250405.txt",
    "gazeteler/20250406.txt",
    "gazeteler/20250407.txt",
    "gazeteler/20250408.txt",
    "gazeteler/20250409.txt",
    "gazeteler/20250410.txt",
    "gazeteler/20250411.txt",
    "gazeteler/20250412.txt",
    "gazeteler/20250413.txt",
]


def clean_text(text):
    """Temel metin temizleme fonksiyonu"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def save_removed_items(removed_items):
    """Çıkarılan öğeleri kaydet"""
    with open("temizlenmis_veriler/removed_items.txt", 'w', encoding='utf-8') as f:
        f.write("=== Çıkarılan Noktalama İşaretleri ===\n")
        f.write(', '.join(removed_items['punctuation']) + "\n\n")
        f.write("=== Çıkarılan Sayılar ===\n")
        f.write(', '.join(removed_items['numbers']) + "\n\n")
        f.write("=== Çıkarılan Stopwords ===\n")
        f.write(', '.join(removed_items['stopwords']) + "\n")


def apply_zipfs_law(words, output_name, output_prefix):
    """Zipf yasası analizi uygula ve CSV'ye ekle"""
    word_counts = Counter(words)
    most_common = word_counts.most_common(1000)

    # Zipf verilerini DataFrame'e çevir
    zipf_df = pd.DataFrame(most_common, columns=['word', 'frequency'])
    zipf_df['rank'] = zipf_df['frequency'].rank(ascending=False, method='min')

    # Zipf grafiği oluştur
    ranks = np.arange(1, len(most_common) + 1)
    frequencies = [count for word, count in most_common]

    plt.figure(figsize=(12, 6))
    plt.loglog(ranks, frequencies, 'b-', marker='o', markersize=3)
    plt.title(f'Zipf Yasası - {output_prefix.capitalize()} Kelime Frekans Dağılımı')
    plt.xlabel('Rank (Log)')
    plt.ylabel('Frekans (Log)')
    plt.grid(True, which="both", ls="-")
    plt.savefig(f"zipf_analizi/{output_prefix}_zipf_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    return zipf_df


def process_and_save_data():
    """Tüm verileri işle ve kaydet"""
    all_lemmatized = []
    all_stemmed = []
    all_words = []
    removed_items = {
        'punctuation': set(),
        'numbers': set(),
        'stopwords': set()
    }

    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()
                cleaned_text = clean_text(text)

                # Çıkarılan öğeleri topla
                removed_items['punctuation'].update(re.findall(r'[^\w\s]', text.lower()))
                removed_items['numbers'].update(re.findall(r'\d+', text))

                words = nltk.word_tokenize(cleaned_text)
                removed_stopwords = [word for word in words if word in stop_words]
                removed_items['stopwords'].update(removed_stopwords)

                # Stopwords'leri kaldır
                filtered_words = [word for word in words if word not in stop_words]
                all_words.extend(filtered_words)

                # Lemmatization ve Stemming uygula
                all_lemmatized.extend([lemmatizer.lemmatize(word) for word in filtered_words])
                all_stemmed.extend([stemmer.stem(word) for word in filtered_words])

            print(f"Başarıyla işlendi: {os.path.basename(txt_file)}")
        except Exception as e:
            print(f"Hata! {txt_file}: {str(e)}")

    # Çıkarılan öğeleri kaydet
    save_removed_items(removed_items)

    # Zipf analizlerini yap ve CSV'lere ekle
    lemmatized_zipf = apply_zipfs_law(all_lemmatized, "Lemmatization", "lemmatized")
    stemmed_zipf = apply_zipfs_law(all_stemmed, "Stemming", "stemmed")

    # Nihai CSV'leri oluştur
    lemmatized_df = pd.DataFrame({
        'word': all_lemmatized,
        'frequency': [all_lemmatized.count(word) for word in all_lemmatized],
        'rank': lemmatized_zipf.set_index('word')['rank'].reindex(all_lemmatized).values
    })

    stemmed_df = pd.DataFrame({
        'word': all_stemmed,
        'frequency': [all_stemmed.count(word) for word in all_stemmed],
        'rank': stemmed_zipf.set_index('word')['rank'].reindex(all_stemmed).values
    })

    # CSV'leri kaydet
    lemmatized_df.to_csv("processed_data/Lemmatization.csv", index=False, encoding='utf-8')
    stemmed_df.to_csv("processed_data/Stemming.csv", index=False, encoding='utf-8')

    # Genel Zipf analizi
    apply_zipfs_law(all_words, "tum_metinler", "genel")

    return all_lemmatized, all_stemmed


# Parametre setleri
param_sets = [
    {'algo': 'cbow', 'window': 2, 'dim': 100},
    {'algo': 'skipgram', 'window': 2, 'dim': 100},
    {'algo': 'cbow', 'window': 4, 'dim': 100},
    {'algo': 'skipgram', 'window': 4, 'dim': 100},
    {'algo': 'cbow', 'window': 2, 'dim': 300},
    {'algo': 'skipgram', 'window': 2, 'dim': 300},
    {'algo': 'cbow', 'window': 4, 'dim': 300},
    {'algo': 'skipgram', 'window': 4, 'dim': 300}
]


def train_word2vec(sentences, params, model_type):
    model = Word2Vec(
        sentences=[sentences],
        sg=1 if params['algo'] == 'skipgram' else 0,
        window=params['window'],
        vector_size=params['dim'],
        min_count=2,
        workers=4,
        epochs=10
    )

    filename = f"word2vec_{model_type}_{params['algo']}_win{params['window']}_dim{params['dim']}.model"
    save_path = os.path.join("word2vec_models", model_type, filename)

    model.save(save_path)
    print(f"Model kaydedildi: {save_path}")
    return model


def main():
    print("Veri yükleme ve ön işleme başlıyor...")
    lemmatized_data, stemmed_data = process_and_save_data()

    print("\nLemmatized modeller eğitiliyor (8 model)...")
    for params in param_sets:
        train_word2vec(lemmatized_data, params, "lemmatized")

    print("\nStemmed modeller eğitiliyor (8 model)...")
    for params in param_sets:
        train_word2vec(stemmed_data, params, "stemmed")

    print("\nTüm işlemler tamamlandı!")
    print("Çıktılar:")
    print("- word2vec_models/: Eğitilmiş Word2Vec modelleri")
    print("- zipf_analizi/: Zipf yasası analiz sonuçları ve grafikler")
    print("- temizlenmis_veriler/: Çıkarılan öğelerin kaydı")
    print("- processed_data/Lemmatization.csv: Lemmatized kelimeler (frekans ve rank bilgileriyle)")
    print("- processed_data/Stemming.csv: Stemmed kelimeler (frekans ve rank bilgileriyle)")


if __name__ == "__main__":
    main()