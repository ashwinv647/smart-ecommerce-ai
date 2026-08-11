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

## Roadmap: extending to the full platform

### Phase 1 (done tonight) — AI/ML core
- [x] Data cleaning + feature engineering
- [x] Semantic search via embeddings + FAISS
- [x] Item-to-item recommendations
- [x] Price prediction model
- [x] FastAPI service wrapping all three

### Phase 2 — Upgrade the AI layer (2-3 evenings)
- [ ] Swap TF-IDF/SVD embeddings for a transformer model (sentence-transformers or
      OpenAI/Anthropic embeddings API) for better semantic quality
- [ ] Add a product **auto-categorization classifier** (text → category/subcategory)
      to handle new listings with missing/wrong categories
- [ ] Add **review/seller anomaly detection** — flag suspicious rating patterns
      (e.g. high rating_count with implausible rating consistency)
- [ ] Add a **RAG shopping assistant** — LLM + the existing FAISS index, so users
      can ask "best budget running shoes under 2000 with good reviews" in natural
      language and get a generated answer + product cards

### Phase 3 — Data Engineering layer (1 week)
- [ ] Move from parquet file to a real warehouse (Postgres/Snowflake) with a
      star schema: `fact_listings` + `dim_category`, `dim_seller`, `dim_brand`
- [ ] Orchestrate the pipeline with Airflow or Prefect (daily refresh simulation)
- [ ] Add data quality checks (Great Expectations or custom) — schema validation,
      null checks, referential integrity, anomaly flags on ingest
- [ ] Add a **streaming layer** (Kafka) simulating live price/stock updates,
      with the index/model refreshed incrementally

### Phase 4 — Full-stack + deployment (1 week)
- [ ] React/Streamlit frontend: search bar, filters, product cards, chat assistant
- [ ] Analytics dashboard: category trends, price distributions, seller quality
      scorecards (built on the warehouse from Phase 3)
- [ ] Containerize (Docker) and deploy (Fly.io/Railway/AWS) with CI/CD
- [ ] Add monitoring: model drift on price predictions, search latency, API metrics

### Stretch goals
- A/B test different embedding models or ranking strategies
- Personalization: simulate user click/purchase history, add collaborative filtering
- Multi-modal search: embed `image_url` product images too, allow image-based search
