"""
preprocess.py
-------------
Data loading and cleaning utilities for the AI Book Recommendation
System, built for the real-world Kaggle **Book Recommendation Dataset**
(arashnic/book-recommendation-dataset):
https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

Expects exactly these files inside dataset/, unmodified filenames:

    dataset/Books.csv
    dataset/Ratings.csv
    dataset/Users.csv

Source columns (as shipped by Kaggle):
    Books.csv   : ISBN, Book-Title, Book-Author, Year-Of-Publication,
                  Publisher, Image-URL-S, Image-URL-M, Image-URL-L
    Ratings.csv : User-ID, ISBN, Book-Rating   (0 = implicit / no rating given)
    Users.csv   : User-ID, Location, Age

Everything is merged and de-duplicated on ISBN, which is the natural
join key across all three files.
"""

import os
import datetime
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

BOOKS_FILE = "Books.csv"
RATINGS_FILE = "Ratings.csv"
USERS_FILE = "Users.csv"

MIN_VALID_RATING = 1          # Book-Crossing uses 0 for "no explicit rating"
MAX_VALID_RATING = 10
MIN_VALID_YEAR = 1450         # roughly the earliest printed books
CURRENT_YEAR = datetime.datetime.now().year

DEFAULT_COVER = "https://via.placeholder.com/128x196.png?text=No+Cover"


# --------------------------------------------------------------------------
# Raw loading — handles the encoding / malformed-row issues that are
# common in the raw Book-Crossing export.
# --------------------------------------------------------------------------
def _read_csv_robust(path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV file while transparently handling encoding issues
    (mixed Latin-1/UTF-8/Windows-1252 bytes) and a handful of malformed
    rows, both of which are well known quirks of the raw Book-Crossing
    export that this Kaggle dataset is built from."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required dataset file not found: {path}\n"
            "Download it from "
            "https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset "
            "and place Books.csv, Ratings.csv and Users.csv inside the dataset/ folder "
            "(filenames must match exactly)."
        )

    last_error = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                sep=",",
                quotechar='"',
                escapechar="\\",
                on_bad_lines="skip",
                low_memory=False,
                **kwargs,
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Could not read {path} with any supported encoding: {last_error}")


def load_raw_data(dataset_dir: str = DATASET_DIR):
    """Load the raw Books, Ratings, and Users CSVs exactly as provided by Kaggle."""
    books_path = os.path.join(dataset_dir, BOOKS_FILE)
    ratings_path = os.path.join(dataset_dir, RATINGS_FILE)
    users_path = os.path.join(dataset_dir, USERS_FILE)

    books = _read_csv_robust(books_path)
    ratings = _read_csv_robust(ratings_path)
    users = _read_csv_robust(users_path)

    return books, ratings, users


def _normalize_isbn(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Books.csv dataframe and normalize it to the
    internal schema used across the app."""
    books = books.copy()

    books = books.rename(columns={
        "ISBN": "isbn",
        "Book-Title": "title",
        "Book-Author": "author",
        "Year-Of-Publication": "year",
        "Publisher": "publisher",
        "Image-URL-S": "image_url_s",
        "Image-URL-M": "image_url_m",
        "Image-URL-L": "image_url_l",
    })

    missing_required = [c for c in ("isbn", "title", "author") if c not in books.columns]
    if missing_required:
        raise ValueError(
            f"Books.csv is missing required column(s): {missing_required}. "
            "Make sure you downloaded the correct Kaggle dataset "
            "(arashnic/book-recommendation-dataset) without renaming its columns."
        )
    for optional in ("year", "publisher", "image_url_s", "image_url_m", "image_url_l"):
        if optional not in books.columns:
            books[optional] = np.nan

    # Missing values: rows with no ISBN/title/author carry nothing usable
    books = books.dropna(subset=["isbn", "title", "author"], how="any")
    books["isbn"] = _normalize_isbn(books["isbn"])

    # Duplicate books: exact duplicate rows, then duplicate ISBNs (the join key)
    books = books.drop_duplicates()
    books = books.drop_duplicates(subset=["isbn"], keep="first")

    # Missing authors / publishers
    books["title"] = books["title"].astype(str).str.strip()
    books["author"] = books["author"].fillna("Unknown Author").astype(str).str.strip()
    books["author"] = books["author"].replace("", "Unknown Author")
    books["publisher"] = books["publisher"].fillna("Unknown Publisher").astype(str).str.strip()
    books["publisher"] = books["publisher"].replace("", "Unknown Publisher")

    # Invalid publication years (Book-Crossing has years like 0, 2038, 9999...)
    books["year"] = pd.to_numeric(books["year"], errors="coerce")
    invalid_year = (books["year"] < MIN_VALID_YEAR) | (books["year"] > CURRENT_YEAR + 1)
    books.loc[invalid_year, "year"] = np.nan
    books["year"] = books["year"].astype("Int64")  # nullable int -> keeps unknown years as <NA>

    # Missing image URLs: prefer Image-URL-M, fall back to L, then S, then a neutral placeholder
    for col in ("image_url_s", "image_url_m", "image_url_l"):
        books[col] = books[col].astype(str).str.strip()
        books[col] = books[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    books["image_url"] = (
        books["image_url_m"]
        .fillna(books["image_url_l"])
        .fillna(books["image_url_s"])
        .fillna(DEFAULT_COVER)
    )

    books = books.reset_index(drop=True)
    return books


def clean_ratings(ratings: pd.DataFrame, valid_isbns=None, valid_user_ids=None) -> pd.DataFrame:
    """Clean the raw Ratings.csv dataframe."""
    ratings = ratings.copy()
    ratings = ratings.rename(columns={"User-ID": "user_id", "ISBN": "isbn", "Book-Rating": "rating"})

    missing_required = [c for c in ("user_id", "isbn", "rating") if c not in ratings.columns]
    if missing_required:
        raise ValueError(f"Ratings.csv is missing required column(s): {missing_required}.")

    ratings = ratings.dropna(subset=["user_id", "isbn", "rating"])
    ratings["isbn"] = _normalize_isbn(ratings["isbn"])

    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "rating"])
    ratings["user_id"] = ratings["user_id"].astype(int)

    # Book-Crossing uses 0 to mean "no explicit rating given" (implicit interaction).
    # Keep only explicit ratings (1-10) so collaborative filtering learns real signal.
    ratings = ratings[(ratings["rating"] >= MIN_VALID_RATING) & (ratings["rating"] <= MAX_VALID_RATING)]
    ratings["rating"] = ratings["rating"].astype(float)

    # Duplicate ratings: same (user, book) rated more than once -> keep the latest occurrence
    ratings = ratings.drop_duplicates(subset=["user_id", "isbn"], keep="last")

    if valid_isbns is not None:
        ratings = ratings[ratings["isbn"].isin(valid_isbns)]
    if valid_user_ids is not None:
        ratings = ratings[ratings["user_id"].isin(valid_user_ids)]

    ratings = ratings.reset_index(drop=True)
    return ratings


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Users.csv dataframe."""
    users = users.copy()
    users = users.rename(columns={"User-ID": "user_id", "Location": "location", "Age": "age"})

    if "user_id" not in users.columns:
        raise ValueError("Users.csv is missing the required 'User-ID' column.")
    if "location" not in users.columns:
        users["location"] = np.nan
    if "age" not in users.columns:
        users["age"] = np.nan

    users = users.dropna(subset=["user_id"])
    users["user_id"] = pd.to_numeric(users["user_id"], errors="coerce")
    users = users.dropna(subset=["user_id"])
    users["user_id"] = users["user_id"].astype(int)

    # Duplicate users
    users = users.drop_duplicates(subset=["user_id"], keep="first")

    # Missing values
    users["location"] = users["location"].fillna("Unknown").astype(str).str.strip()
    users["location"] = users["location"].replace("", "Unknown")

    # Invalid ages -> unknown rather than dropping the user (they may still have valid ratings)
    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    invalid_age = (users["age"] < 5) | (users["age"] > 100)
    users.loc[invalid_age, "age"] = np.nan
    users["age"] = users["age"].astype("Int64")

    users = users.reset_index(drop=True)
    return users


def compute_book_stats(books: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Attach average rating (1-10 scale) and rating count to each book, keyed on ISBN."""
    stats = ratings.groupby("isbn")["rating"].agg(["mean", "count"]).reset_index()
    stats.columns = ["isbn", "avg_rating", "rating_count"]

    merged = books.merge(stats, on="isbn", how="left")
    merged["avg_rating"] = merged["avg_rating"].fillna(0).round(2)
    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
    merged = merged.reset_index(drop=True)
    return merged


def build_content_features(books: pd.DataFrame) -> pd.Series:
    """Combine title, author and publisher into one text field used as
    TF-IDF input. The real dataset has no genre or description column,
    so the content model is built only from these real fields (title
    and author weighted x2 as the strongest similarity signals)."""
    title = books["title"].astype(str)
    author = books["author"].astype(str)
    publisher = books["publisher"].astype(str)
    combined = title + " " + title + " " + author + " " + author + " " + publisher
    return combined.str.lower()


def load_and_clean_all(dataset_dir: str = DATASET_DIR):
    """Full preprocessing pipeline. Returns clean books (with rating
    stats), clean ratings, and clean users dataframes, all merged/keyed
    on ISBN (books & ratings) and User-ID (users & ratings)."""
    books, ratings, users = load_raw_data(dataset_dir)

    books = clean_books(books)
    users = clean_users(users)
    ratings = clean_ratings(
        ratings,
        valid_isbns=set(books["isbn"]),
        valid_user_ids=set(users["user_id"]),
    )

    books_with_stats = compute_book_stats(books, ratings)

    return books_with_stats, ratings, users


if __name__ == "__main__":
    b, r, u = load_and_clean_all()
    print(f"Clean books:   {len(b):,}")
    print(f"Clean ratings: {len(r):,}")
    print(f"Clean users:   {len(u):,}")
