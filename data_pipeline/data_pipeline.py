"""
Module 1 - Data Pipeline
Books to Scrape -> cleaning -> fixed GBP/INR conversion -> SQLite -> SQL/pandas checks.

Run:
    python data_pipeline.py

The script is intentionally self-contained so the SQLite database can be
regenerated from scratch without committing generated data.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from statistics import median
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"
FIXED_GBP_TO_INR = 105.50
MIN_ROWS = 60
MIN_CATEGORIES = 3

# These three categories currently contain 11 + 32 + 26 = 69 books.
CATEGORY_URLS = {
    "Travel": urljoin(BASE_URL, "catalogue/category/books/travel_2/index.html"),
    "Mystery": urljoin(BASE_URL, "catalogue/category/books/mystery_3/index.html"),
    "Historical Fiction": urljoin(
        BASE_URL, "catalogue/category/books/historical-fiction_4/index.html"
    ),
}

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "books.db"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ZeptoCapstoneDataPipeline/1.0)"
}


def fetch(url: str) -> str:
    """GET a page and fail with a useful error on non-2xx responses."""
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return response.text


def parse_rating(class_list: list[str]) -> int | None:
    """Convert One/Two/...Five to 1..5."""
    mapping = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }
    for class_name in class_list:
        if class_name in mapping:
            return mapping[class_name]
    return None


def parse_price(text: str) -> float | None:
    """Extract the numeric GBP amount from text such as '£45.17'."""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text.replace(",", ""))
    return float(match.group(1)) if match else None


def parse_page(
    html: str, category: str, current_url: str
) -> tuple[list[dict], str | None]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

    for product in soup.select("article.product_pod"):
        title_tag = product.select_one("h3 a")
        price_tag = product.select_one(".price_color")
        rating_tag = product.select_one(".star-rating")
        availability_tag = product.select_one(".availability")

        title = title_tag.get("title", "").strip() if title_tag else ""
        price_gbp = parse_price(price_tag.get_text(" ", strip=True)) if price_tag else None
        rating = (
            parse_rating(rating_tag.get("class", []))
            if rating_tag
            else None
        )
        availability = (
            availability_tag.get_text(" ", strip=True)
            if availability_tag
            else ""
        )

        rows.append(
            {
                "title": title,
                "price_gbp_raw": price_tag.get_text(" ", strip=True)
                if price_tag
                else "",
                "price_gbp": price_gbp,
                "star_rating_raw": (
                    " ".join(rating_tag.get("class", []))
                    if rating_tag
                    else ""
                ),
                "rating": rating,
                "availability": availability,
                "category": category,
            }
        )

    next_link = soup.select_one("li.next a")
    next_url = urljoin(current_url, next_link["href"]) if next_link else None
    return rows, next_url


def scrape_categories() -> pd.DataFrame:
    """Scrape all books in the three selected categories, following pagination."""
    all_rows: list[dict] = []

    for category, start_url in CATEGORY_URLS.items():
        url: str | None = start_url
        pages = 0

        while url:
            pages += 1
            html = fetch(url)
            rows, next_url = parse_page(html, category, url)
            all_rows.extend(rows)
            url = next_url

        print(f"Scraped {category}: {len([r for r in all_rows if r['category'] == category])} books across {pages} page(s).")

    df = pd.DataFrame(all_rows)

    # Defensive duplicate handling: one book/category pair is one catalog record.
    df = df.drop_duplicates(subset=["title", "category"], keep="first").reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw fields and apply the required median-imputation policy."""
    df = df.copy()

    # Drop rows where the two categorical identifiers cannot be recovered.
    df["title"] = df["title"].astype("string").str.strip()
    df["category"] = df["category"].astype("string").str.strip()
    before = len(df)
    df = df[df["title"].notna() & df["title"].ne("")].copy()
    print(f"Dropped {before - len(df)} rows with unusable title values.")

    # Numeric fields: convert invalid values to NaN, then median-impute.
    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    for column in ["price_gbp", "rating"]:
        if df[column].isna().any():
            if df[column].dropna().empty:
                raise ValueError(f"Cannot median-impute {column}: all values failed to parse.")
            fill_value = df[column].median()
            df[column] = df[column].fillna(fill_value)
            print(f"Median-imputed {column} with {fill_value}.")

    # Rating is required to be an integer 1-5.
    df["rating"] = df["rating"].round().clip(1, 5).astype(int)

    # Availability text -> boolean.
    df["in_stock"] = (
        df["availability"]
        .astype("string")
        .str.contains("In stock", case=False, na=False)
    )

    # Required fixed project-defined conversion. No API or live FX lookup.
    df["price_inr"] = (df["price_gbp"] * FIXED_GBP_TO_INR).round(2)

    # Keep only the normalized fields required by the specification.
    return df[
        ["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]
    ].reset_index(drop=True)


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        DROP TABLE IF EXISTS books;
        DROP TABLE IF EXISTS categories;

        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        );
        """
    )


def load_database(df: pd.DataFrame, db_path: Path) -> None:
    """Load normalized data into a two-table SQLite schema."""
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        create_schema(conn)

        categories = (
            df[["category"]]
            .drop_duplicates()
            .sort_values("category")
            .reset_index(drop=True)
        )
        categories.columns = ["category_name"]
        categories.to_sql("categories", conn, if_exists="append", index=False)

        category_lookup = pd.read_sql(
            "SELECT category_id, category_name FROM categories", conn
        )
        book_rows = df.merge(
            category_lookup,
            left_on="category",
            right_on="category_name",
            how="left",
        )[
            ["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]
        ].copy()
        book_rows["in_stock"] = book_rows["in_stock"].astype(int)
        book_rows.to_sql("books", conn, if_exists="append", index=False)

        # Verify relational integrity.
        orphan_count = pd.read_sql(
            """
            SELECT COUNT(*) AS orphan_count
            FROM books b
            LEFT JOIN categories c ON b.category_id = c.category_id
            WHERE c.category_id IS NULL
            """,
            conn,
        ).iloc[0]["orphan_count"]

        if orphan_count != 0:
            raise AssertionError("Foreign-key integrity check failed.")

        print(f"SQLite database created: {db_path}")
        print(f"Loaded {len(book_rows)} book rows and {len(categories)} categories.")


QUERIES = {
    "01_select_all": """
        SELECT book_id, title, price_gbp, price_inr, rating, in_stock
        FROM books
        ORDER BY book_id
        LIMIT 10;
    """,
    "02_where_in_stock": """
        SELECT title, rating, price_inr
        FROM books
        WHERE in_stock = 1 AND rating >= 4
        ORDER BY rating DESC, price_inr ASC;
    """,
    "03_order_by_price": """
        SELECT title, price_gbp, price_inr
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,
    "04_limit": """
        SELECT title, rating
        FROM books
        ORDER BY rating DESC, title
        LIMIT 5;
    """,
    "05_distinct": """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
    """,
    "06_in_between": """
        SELECT title, category_id, price_gbp
        FROM books
        WHERE category_id IN (1, 2, 3)
          AND price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
    """,
    "07_join_top_10_per_category": """
        SELECT
            category_name,
            title,
            rating,
            price_gbp,
            price_inr,
            in_stock
        FROM (
            SELECT
                c.category_name,
                b.title,
                b.rating,
                b.price_gbp,
                b.price_inr,
                b.in_stock,
                ROW_NUMBER() OVER (
                    PARTITION BY c.category_name
                    ORDER BY b.rating DESC, b.price_gbp ASC, b.title
                ) AS rn
            FROM books AS b
            JOIN categories AS c
              ON b.category_id = c.category_id
        )
        WHERE rn <= 10
        ORDER BY category_name, rating DESC, price_gbp ASC, title;
    """,
}


def run_sql_queries(db_path: Path) -> dict[str, pd.DataFrame]:
    """Execute and print all required SQL demonstrations."""
    results: dict[str, pd.DataFrame] = {}

    with sqlite3.connect(db_path) as conn, open(
        OUTPUT_DIR / "sql_query_output.txt", "w", encoding="utf-8"
    ) as log:
        for name, sql in QUERIES.items():
            df = pd.read_sql_query(sql, conn)
            results[name] = df

            print(f"\n=== {name} ===")
            print(df.to_string(index=False))

            log.write(f"\n=== {name} ===\n")
            log.write(df.to_string(index=False))
            log.write("\n")

    return results


def compare_sql_join_with_pandas(db_path: Path) -> None:
    """Read the join inputs with pd.read_sql and reproduce the join with pd.merge."""
    with sqlite3.connect(db_path) as conn:
        sql_join = pd.read_sql_query(
            """
            SELECT
                c.category_name,
                b.title,
                b.rating,
                b.price_gbp,
                b.price_inr,
                b.in_stock
            FROM books AS b
            JOIN categories AS c
              ON b.category_id = c.category_id
            ORDER BY c.category_name, b.title;
            """,
            conn,
        )

        books_df = pd.read_sql(
            """
            SELECT title, price_gbp, price_inr, rating, in_stock, category_id
            FROM books;
            """,
            conn,
        )
        categories_df = pd.read_sql(
            """
            SELECT category_id, category_name
            FROM categories;
            """,
            conn,
        )

    pandas_join = (
        books_df.merge(categories_df, on="category_id", how="inner")
        [["category_name", "title", "rating", "price_gbp", "price_inr", "in_stock"]]
        .sort_values(["category_name", "title"])
        .reset_index(drop=True)
    )
    sql_join = sql_join.reset_index(drop=True)

    # SQLite returns 0/1 while pandas may use bool; normalize before comparison.
    sql_join["in_stock"] = sql_join["in_stock"].astype(bool)
    pandas_join["in_stock"] = pandas_join["in_stock"].astype(bool)

    equivalent = sql_join.equals(pandas_join)
    if not equivalent:
        raise AssertionError("SQL JOIN and pandas merge results do not match.")

    comparison = pd.concat(
        [
            sql_join.add_prefix("SQL_"),
            pandas_join.add_prefix("PANDAS_"),
        ],
        axis=1,
    )

    print("\n=== SQL JOIN vs pandas.merge (side-by-side) ===")
    print(comparison.to_string(index=False))
    print(f"\nEquivalent output: {equivalent}")

    comparison.to_csv(OUTPUT_DIR / "join_sql_vs_pandas.csv", index=False)

    with open(
        OUTPUT_DIR / "join_comparison.txt", "w", encoding="utf-8"
    ) as f:
        f.write("SQL JOIN vs pandas.merge (side-by-side)\n\n")
        f.write(comparison.to_string(index=False))
        f.write(f"\n\nEquivalent output: {equivalent}\n")


def validate_acceptance(df: pd.DataFrame, db_path: Path) -> None:
    """Fail loudly if a rubric acceptance criterion is not met."""
    assert len(df) >= MIN_ROWS, f"Need >= {MIN_ROWS} rows; got {len(df)}."
    assert df["category"].nunique() >= MIN_CATEGORIES, (
        f"Need >= {MIN_CATEGORIES} categories; got {df['category'].nunique()}."
    )
    assert pd.api.types.is_float_dtype(df["price_gbp"])
    assert pd.api.types.is_integer_dtype(df["rating"])
    assert df["rating"].between(1, 5).all()
    assert pd.api.types.is_bool_dtype(df["in_stock"])
    assert pd.api.types.is_float_dtype(df["price_inr"])

    expected = (df["price_gbp"] * FIXED_GBP_TO_INR).round(2)
    assert (df["price_inr"] == expected).all()

    assert db_path.exists()

    with sqlite3.connect(db_path) as conn:
        book_count = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        category_count = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        assert book_count == len(df)
        assert category_count == df["category"].nunique()

    print("\nAcceptance checks: PASSED")
    print(f"Rows: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Fixed rate: 1 GBP = {FIXED_GBP_TO_INR:.2f} INR")


def main() -> None:
    print("Starting Module 1 data pipeline...")
    raw_df = scrape_categories()
    print(f"\nRaw scraped rows: {len(raw_df)}")

    clean_df = clean_data(raw_df)
    print(f"Cleaned rows: {len(clean_df)}")

    load_database(clean_df, DB_PATH)
    run_sql_queries(DB_PATH)
    compare_sql_join_with_pandas(DB_PATH)
    validate_acceptance(clean_df, DB_PATH)

    clean_df.to_csv(OUTPUT_DIR / "cleaned_books.csv", index=False)
    print("\nGenerated:")
    print(f"  - {DB_PATH}")
    print(f"  - {OUTPUT_DIR / 'cleaned_books.csv'}")
    print(f"  - {OUTPUT_DIR / 'sql_query_output.txt'}")
    print(f"  - {OUTPUT_DIR / 'join_comparison.txt'}")
    print(f"  - {OUTPUT_DIR / 'join_sql_vs_pandas.csv'}")


if __name__ == "__main__":
    main()
