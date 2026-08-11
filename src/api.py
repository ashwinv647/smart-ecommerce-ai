"""
FastAPI serving layer for the E-Commerce AI Platform.

Endpoints:
  GET  /search?q=...&category=&max_price=&min_rating=&prime_only=
  GET  /similar/{product_id}
  POST /predict-price

Run with:
  uvicorn src.api:app --reload --port 8000
Then visit http://localhost:8000/docs for interactive Swagger UI.
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
from pathlib import Path

from search import ProductSearch
from price_model import predict_price

app = FastAPI(
    title="E-Commerce AI Platform",
    description="Semantic search, recommendations, and price prediction over a product catalog.",
    version="0.1.0",
)

BASE = Path(__file__).parent.parent
searcher = ProductSearch()
price_model = joblib.load(BASE / "models" / "price_model.joblib")


class PriceRequest(BaseModel):
    category: str
    subcategory: str
    brand: str
    seller_type: str
    color: str
    rating: float
    rating_count: int
    seller_rating: float
    stock_quantity: int
    listing_age_days: int
    prime_eligible: bool
    in_stock: bool


@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoints": ["/search", "/similar/{product_id}", "/predict-price", "/docs"],
    }


@app.get("/search")
def search(
    q: str = Query(..., description="Natural language search query"),
    k: int = Query(10, ge=1, le=50),
    category: str | None = None,
    max_price: float | None = None,
    min_rating: float | None = None,
    prime_only: bool = False,
):
    results = searcher.search(
        q, k=k, category=category, max_price=max_price,
        min_rating=min_rating, prime_only=prime_only,
    )
    return {"query": q, "count": len(results), "results": results.to_dict(orient="records")}


@app.get("/similar/{product_id}")
def similar(product_id: str, k: int = Query(10, ge=1, le=50)):
    try:
        results = searcher.similar_products(product_id, k=k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"product_id": product_id, "count": len(results), "results": results.to_dict(orient="records")}


@app.post("/predict-price")
def predict(req: PriceRequest):
    price = predict_price(price_model, req.model_dump())
    return {"predicted_price": round(price, 2), "currency": "INR"}
