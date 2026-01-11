import numpy as np
import re
from collections import Counter
import os

def load_data(data_dir='data/lyrics'):
    """
    Load all .txt files from the specified directory
    """
    import os
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    texts = []
    artists = []
    
    for f in files:
        full_path = os.path.join(data_dir, f)
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read().lower()
            text = re.sub(r'\W+', ' ', text)
            texts.append(text.strip())
            artists.append(f[:-4])  # remove .txt
    
    print(f"Loaded {len(artists)} artist lyrics from {data_dir}")
    return texts, artists

def compute_tfidf(texts, max_vocab=500):
    """
    Compute simple TF-IDF features from list of texts
    Returns: features (np.array), vocabulary (list), artists (list)
    """
    N = len(texts)
    if N == 0:
        return np.array([]), [], []

    # Get most common words
    all_words = ' '.join(texts).split()
    word_counts = Counter(all_words)
    vocab = [word for word, count in word_counts.most_common(max_vocab)]
    vocab_size = len(vocab)

    if vocab_size == 0:
        return np.zeros((N, 0)), [], []

    word_to_idx = {word: i for i, word in enumerate(vocab)}

    # Term Frequency (TF)
    tf = np.zeros((N, vocab_size))
    for i, text in enumerate(texts):
        words = text.split()
        counts = Counter(words)
        total_words = len(words) if len(words) > 0 else 1  # avoid div by zero
        for word, count in counts.items():
            if word in word_to_idx:
                tf[i, word_to_idx[word]] = count / total_words

    # Inverse Document Frequency (IDF)
    df = np.sum(tf > 0, axis=0)                     # document frequency
    idf = np.log(N / (df + 1)) + 1                   # smoothed IDF

    # TF-IDF
    features = tf * idf

    # We need artists — but they're not passed here!
    # Solution A: Return them from load_data() and pass them
    # Solution B (recommended for this project): return artists too
    # But since this function is called after load_data, we usually do:

    # For now we'll just return what we have (you'll pass artists separately)
    return features.astype(np.float32), vocab

