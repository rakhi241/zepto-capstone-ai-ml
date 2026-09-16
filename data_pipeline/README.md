# Module 1 — Data Pipeline

This module implements the capstone data-engineering pipeline:

**scrape → clean → fixed-rate conversion → normalized SQLite storage → SQL queries → pandas validation**

## Data source

The pipeline uses [Books to Scrape](https://books.toscrape.com/), a public scraping-practice website. It requires no login, API key, or paid service.

The selected categories are:

- Travel
- Mystery
- Historical Fiction

The current site contains 11 Travel books, 32 Mystery books, and 26 Historical Fiction books, giving 69 catalog records before any defensive deduplication. The script follows pagination automatically, so it does not rely on manual copy-pasting.

## Fixed conversion rate

The assignment requires the project-defined constant:

**1 GBP = 105.50 INR**

This is deliberately hard-coded in `data_pipeline.py`. No live currency API or date-based lookup is used.

## Setup

From the repository root:

```bash
cd data_pipeline
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run end-to-end

```bash
python data_pipeline.py
```

The script:

1. Scrapes all books in the three selected categories.
2. Extracts title, GBP price, star rating, availability, and category.
3. Cleans price and rating into numeric types.
4. Converts availability into the boolean `in_stock`.
5. Median-imputes numeric values if a row contains an unexpected parse failure.
6. Computes `price_inr` using the required fixed rate.
7. Recreates a normalized SQLite database from scratch.
8. Executes seven SQL demonstrations.
9. Reads SQL results into pandas.
10. Reproduces the relational join with `pd.merge()`.
11. Compares SQL JOIN and pandas merge outputs and fails if they differ.
12. Runs acceptance checks.

## Database schema

### `categories`

| Column | Type | Constraint |
|---|---|---|
| `category_id` | INTEGER | PRIMARY KEY |
| `category_name` | TEXT | NOT NULL, UNIQUE |

### `books`

| Column | Type | Constraint |
|---|---|---|
| `book_id` | INTEGER | PRIMARY KEY |
| `title` | TEXT | NOT NULL |
| `price_gbp` | REAL | NOT NULL |
| `price_inr` | REAL | NOT NULL |
| `rating` | INTEGER | 1–5 CHECK |
| `in_stock` | INTEGER | 0/1 CHECK |
| `category_id` | INTEGER | FOREIGN KEY → `categories.category_id` |

The normalized design avoids repeating category names in every category record and demonstrates a real primary-key/foreign-key relationship.

## SQL demonstrations

The script executes and logs these queries:

1. `SELECT`
2. `WHERE`
3. `ORDER BY`
4. `LIMIT`
5. `DISTINCT`
6. `IN` + `BETWEEN`
7. `JOIN`

Outputs are written to:

- `outputs/sql_query_output.txt`
- `outputs/join_comparison.txt`
- `outputs/join_sql_vs_pandas.csv`

## pandas SQL + merge requirement

The join is performed in two ways:

- SQL: `JOIN categories ... ON books.category_id = categories.category_id`
- pandas: `books_df.merge(categories_df, on="category_id", how="inner")`

Both results are sorted consistently and compared with `DataFrame.equals()`. The script stops with an error if they do not match.

## Generated files

`books.db` is generated from scratch by the script.

`outputs/` contains reproducible query evidence and the cleaned dataset. These are generated artifacts; the source of truth is the Python pipeline.

## Design / parsing decisions

- The scraper follows each category's `next` link instead of hard-coding page counts.
- Rows without a usable title are dropped because a title is the primary business identifier needed for a catalog record.
- Numeric parsing failures are converted to missing values and median-imputed, as required by the specification.
- Ratings are converted from words (`One`–`Five`) to integers 1–5.
- Availability is converted to a boolean based on the presence of `In stock`.
- GBP→INR uses only the required fixed rate of 105.50.
- SQLite is used because it is built into Python, requires no server, and is sufficient to demonstrate relational modeling, foreign keys, SQL querying, and pandas interoperability.
