"""
Data pipeline: load, clean, and validate the product catalog.
This is the seam where a future Airflow/dbt pipeline plugs in —
today it's a function, tomorrow it's a scheduled task reading from
a warehouse table instead of an xlsx file.
"""
import pandas as pd
from pathlib import Path

RAW_PATH = "/mnt/user-data/uploads/amazon_products_sample.xlsx"
CLEAN_PATH = Path(__file__).parent.parent / "data" / "products_clean.parquet"


def load_and_clean(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_excel(raw_path)

    # --- validation / cleaning rules ---
    df["color"] = df["color"].fillna("Unspecified")
    df = df.drop_duplicates(subset="product_id")
    df = df[df["price"] > 0]
    df = df[(df["rating"] >= 0) & (df["rating"] <= 5)]

    # --- feature engineering ---
    df["discount_amount"] = df["mrp"] - df["price"]
    df["price_per_rating_count"] = df["price"] / (df["rating_count"] + 1)
    df["listing_age_days"] = (
        pd.Timestamp.now() - df["listing_date"]
    ).dt.days

    df = df.reset_index(drop=True)
    return df


def save_clean(df: pd.DataFrame, path: Path = CLEAN_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} clean rows -> {path}")


if __name__ == "__main__":
    df = load_and_clean()
    save_clean(df)
    print(df.info())
