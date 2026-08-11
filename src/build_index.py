"""
Build semantic embeddings for the product catalog and index them
with FAISS for fast similarity search.

NOTE ON MODEL CHOICE:
This sandbox has no internet access to Hugging Face, so we use
TF-IDF + Truncated SVD (a classic LSA-style embedding) built with
scikit-learn entirely offline. It captures semantic similarity
surprisingly well for product search.

EXTENSION POINT: in a real deployment with internet access, swap
the `fit_embed()` function below for a transformer encoder, e.g.:

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, normalize_embeddings=True)

Everything downstream (FAISS index, search.py, API) is agnostic to
how the embedding vectors were produced, so this swap is a one-file change.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import faiss

DATA_PATH = Path(__file__).parent.parent / "data" / "products_clean.parquet"
INDEX_PATH = Path(__file__).parent.parent / "models" / "product_index.faiss"
EMB_PATH = Path(__file__).parent.parent / "models" / "product_embeddings.npy"
IDS_PATH = Path(__file__).parent.parent / "models" / "product_ids.npy"
VECTORIZER_PATH = Path(__file__).parent.parent / "models" / "tfidf_vectorizer.joblib"
SVD_PATH = Path(__file__).parent.parent / "models" / "svd_model.joblib"

N_COMPONENTS = 200  # embedding dimensionality


def fit_embed(texts):
    """Fit TF-IDF + SVD on the corpus and return dense normalized embeddings."""
    vectorizer = TfidfVectorizer(
        max_features=20000, stop_words="english", ngram_range=(1, 2), min_df=2
    )
    tfidf = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    embeddings = svd.fit_transform(tfidf)
    embeddings = normalize(embeddings)  # so inner product == cosine similarity

    return embeddings.astype("float32"), vectorizer, svd


def build():
    df = pd.read_parquet(DATA_PATH)
    print(f"Encoding {len(df)} products with TF-IDF + SVD...")

    embeddings, vectorizer, svd = fit_embed(df["Combine_Text"].tolist())

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    np.save(EMB_PATH, embeddings)
    np.save(IDS_PATH, df["product_id"].values)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(svd, SVD_PATH)

    explained = svd.explained_variance_ratio_.sum()
    print(f"Indexed {index.ntotal} vectors, dim={embeddings.shape[1]}")
    print(f"SVD explained variance: {explained:.2%}")
    print(f"Saved index -> {INDEX_PATH}")


if __name__ == "__main__":
    build()
