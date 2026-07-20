"""
helper.py
---------
Miscellaneous helper functions used across the Streamlit application:
trending book computation, search utilities, and small formatting
helpers.
"""

import pandas as pd


def get_top_rated_books(books: pd.DataFrame, min_votes: int = 5, top_n: int = 10) -> pd.DataFrame:
    """Return books with the highest average rating, requiring a
    minimum number of ratings to avoid a single 10/10 vote dominating."""
    eligible = books[books["rating_count"] >= min_votes].copy()
    if eligible.empty:
        eligible = books.copy()
    eligible = eligible.sort_values(
        by=["avg_rating", "rating_count"], ascending=[False, False]
    )
    return eligible.head(top_n).reset_index(drop=True)


def get_most_popular_books(books: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return books with the highest number of ratings (most reviewed)."""
    popular = books.sort_values(by="rating_count", ascending=False)
    return popular.head(top_n).reset_index(drop=True)


def get_most_reviewed_books(books: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Alias for most reviewed / most popular books, kept separate for
    clarity in the UI (may diverge if additional metrics are added)."""
    return get_most_popular_books(books, top_n)


def get_highest_rated_decade_leaders(books: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the single highest-rated book from each publication decade.

    The real Kaggle dataset has no genre field, so this replaces the
    old genre-leaders view with a diversity view across eras instead,
    built entirely from the real Year-Of-Publication column.
    """
    eligible = books[(books["rating_count"] >= 3) & books["year"].notna()].copy()
    if eligible.empty:
        return books.iloc[0:0].copy()

    eligible["decade"] = (eligible["year"].astype(int) // 10 * 10).astype(int)
    idx = eligible.groupby("decade")["avg_rating"].idxmax()
    leaders = eligible.loc[idx].sort_values("decade", ascending=False)
    return leaders.head(top_n).reset_index(drop=True)


def search_books(books: pd.DataFrame, query: str) -> pd.DataFrame:
    """Case-insensitive search across title and author fields."""
    if not query or not query.strip():
        return books.iloc[0:0]

    query = query.strip().lower()
    mask = (
        books["title"].astype(str).str.lower().str.contains(query, na=False, regex=False) |
        books["author"].astype(str).str.lower().str.contains(query, na=False, regex=False)
    )
    return books[mask].reset_index(drop=True)


def format_rating_stars(rating: float, max_rating: float = 10.0) -> str:
    """Convert a numeric rating (on a max_rating scale) into a 5-star string representation."""
    if pd.isna(rating):
        rating = 0
    stars_float = (rating / max_rating) * 5
    full_stars = int(stars_float)
    half_star = 1 if (stars_float - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    return "★" * full_stars + ("½" if half_star else "") + "☆" * empty_stars


def truncate_text(text: str, max_len: int = 160) -> str:
    """Truncate long text for card display."""
    if not isinstance(text, str):
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."
