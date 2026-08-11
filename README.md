# E-Commerce AI Platform

An AI/ML platform over a 10,000-product Amazon-style catalog: semantic search,
item-to-item recommendations, and price prediction, served via a FastAPI backend.

Built as an evening MVP (AI/ML layer), scoped to extend into a full
Data Engineering + AI + full-stack portfolio project.

## What's built tonight

```
ecommerce_ai/
├── data/
│   └── products_clean.parquet       # cleaned catalog (10,000 rows)
├── models/
│   ├── product_index.faiss          # vector index for semantic search
│   ├── tfidf_vectorizer.joblib       # TF-IDF vectorizer
│   ├── svd_model.joblib              # SVD embedding model (200-dim, 86% variance)
│   └── price_model.joblib            # Gradient Boosting price predictor
├── src/
│   ├── data_pipeline.py              # load, clean, feature-engineer
│   ├── build_index.py                # build embeddings + FAISS index
│   ├── search.py                     # semantic search + recommendations
│   ├── price_model.py                # price prediction training + inference
│   └── api.py                        # FastAPI service (3 endpoints)
└── requirements.txt
```

### Run it

```bash
pip install -r requirements.txt
python src/data_pipeline.py      # clean the raw catalog
python src/build_index.py        # build embeddings + index
python src/price_model.py        # train the price model
uvicorn src.api:app --reload --port 8000
# visit http://localhost:8000/docs
```

### What each piece does

1. **Semantic search** (`/search?q=...`) — free-text query → TF-IDF+SVD embedding
   → FAISS cosine similarity → ranked results, with optional filters
   (category, max price, min rating, prime-only).
2. **Recommendations** (`/similar/{product_id}`) — item-to-item similarity using
   the same embedding space. Zero extra training required.
3. **Price prediction** (`POST /predict-price`) — Gradient Boosting regressor on
   category/brand/rating/seller features, trained in log-price space.
   Test performance: **R² = 0.946** (log-space), MAE ≈ ₹12.5k.

### Model choice note

Embeddings use TF-IDF + Truncated SVD (scikit-learn), not a transformer model —
this sandbox has no internet access to Hugging Face. `build_index.py` documents
exactly where to swap in `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) once
deployed somewhere with internet access; every downstream component (FAISS index,
search, API) is agnostic to how the vectors were produced.

---

