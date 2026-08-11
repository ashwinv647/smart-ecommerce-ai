"""
Price prediction model: given product attributes, predict a fair market price.
Useful for: sellers pricing new listings, flagging mispriced items,
or estimating price for a product description with no price yet.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE = Path(__file__).parent.parent
DATA_PATH = BASE / "data" / "products_clean.parquet"
MODEL_PATH = BASE / "models" / "price_model.joblib"

CATEGORICAL = ["category", "subcategory", "brand", "seller_type", "color"]
NUMERIC = ["rating", "rating_count", "seller_rating", "stock_quantity", "listing_age_days"]
BOOLEAN = ["prime_eligible", "in_stock"]
TARGET = "price"


def train():
    df = pd.read_parquet(DATA_PATH)
    df["log_price"] = np.log1p(df[TARGET])  # price is right-skewed; model in log-space

    features = CATEGORICAL + NUMERIC + BOOLEAN
    X = df[features]
    y = df["log_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
        ("bool", "passthrough", BOOLEAN),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("gbr", GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
        )),
    ])

    model.fit(X_train, y_train)

    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, preds)
    r2 = r2_score(y_test, preds_log)
    mape = np.mean(np.abs((actual - preds) / actual)) * 100

    print(f"Test MAE: ₹{mae:,.0f}")
    print(f"Test MAPE: {mape:.1f}%")
    print(f"Test R² (log-space): {r2:.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    return model


def predict_price(model, product_attrs: dict) -> float:
    """product_attrs must contain all CATEGORICAL + NUMERIC + BOOLEAN keys."""
    X = pd.DataFrame([product_attrs])
    log_pred = model.predict(X)[0]
    return float(np.expm1(log_pred))


if __name__ == "__main__":
    model = train()

    sample = {
        "category": "Electronics", "subcategory": "Smartphones", "brand": "Apple",
        "seller_type": "Official Store", "color": "Black",
        "rating": 4.5, "rating_count": 500, "seller_rating": 4.7,
        "stock_quantity": 100, "listing_age_days": 30,
        "prime_eligible": True, "in_stock": True,
    }
    print(f"\nSample prediction: ₹{predict_price(model, sample):,.0f}")
