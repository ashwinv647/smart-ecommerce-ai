"""
Semantic search over the product catalog.
Embeds a free-text query the same way the catalog was embedded,
then finds nearest neighbors in the FAISS index.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import faiss

BASE = Path(__file__).parent.parent
DATA_PATH = BASE / "data" / "products_clean.parquet"
INDEX_PATH = BASE / "models" / "product_index.faiss"
IDS_PATH = BASE / "models" / "product_ids.npy"
VECTORIZER_PATH = BASE / "models" / "tfidf_vectorizer.joblib"
SVD_PATH = BASE / "models" / "svd_model.joblib"


class ProductSearch:
    def __init__(self):
        self.df = pd.read_parquet(DATA_PATH).set_index("product_id", drop=False)
        self.index = faiss.read_index(str(INDEX_PATH))
        self.ids = np.load(IDS_PATH, allow_pickle=True)
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.svd = joblib.load(SVD_PATH)

    def _embed_query(self, text: str) -> np.ndarray:
        tfidf = self.vectorizer.transform([text])
        vec = self.svd.transform(tfidf)
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-10)
        return vec.astype("float32")

    def search(
        self,
        query: str,
        k: int = 10,
        category: str | None = None,
        max_price: float | None = None,
        min_rating: float | None = None,
        prime_only: bool = False,
    ) -> pd.DataFrame:
        # over-fetch then filter, so filters don't starve results
        fetch_k = max(k * 5, 50)
        qvec = self._embed_query(query)
        scores, idxs = self.index.search(qvec, fetch_k)

        result_ids = self.ids[idxs[0]]
        results = self.df.loc[result_ids].copy()
        results["similarity"] = scores[0]

        if category:
            results = results[results["category"].str.lower() == category.lower()]
        if max_price:
            results = results[results["price"] <= max_price]
        if min_rating:
            results = results[results["rating"] >= min_rating]
        if prime_only:
            results = results[results["prime_eligible"]]

        cols = [
            "product_id", "product_name", "brand", "category", "price",
            "rating", "rating_count", "prime_eligible", "similarity",
        ]
        return results[cols].head(k).reset_index(drop=True)

    def similar_products(self, product_id: str, k: int = 10) -> pd.DataFrame:
        """Find products similar to a given product (item-to-item recommendations)."""
        pos = np.where(self.ids == product_id)[0]
        if len(pos) == 0:
            raise ValueError(f"product_id {product_id} not found")
        pos = pos[0]

        qvec = self.index.reconstruct(int(pos)).reshape(1, -1)
        scores, idxs = self.index.search(qvec, k + 1)  # +1 to drop itself

        result_ids = self.ids[idxs[0]]
        results = self.df.loc[result_ids].copy()
        results["similarity"] = scores[0]
        results = results[results["product_id"] != product_id]

        cols = [
            "product_id", "product_name", "brand", "category", "price",
            "rating", "similarity",
        ]
        return results[cols].head(k).reset_index(drop=True)


if __name__ == "__main__":
    ps = ProductSearch()

    print("=== Semantic search: 'wireless earbuds for running' ===")
    print(ps.search("wireless earbuds for running", k=5).to_string(index=False))

    print("\n=== Filtered search: budget smartphones under 20000, rating>=4 ===")
    print(
        ps.search(
            "smartphone", k=5, category="Electronics", max_price=20000, min_rating=4.0
        ).to_string(index=False)
    )

    print("\n=== Similar products to first result ===")
    first_id = ps.search("running shoes", k=1)["product_id"].iloc[0]
    print(f"Base product: {first_id}")
    print(ps.similar_products(first_id, k=5).to_string(index=False))
