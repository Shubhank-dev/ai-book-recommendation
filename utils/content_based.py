"""
content_based.py
-----------------
Content-Based Filtering engine for the AI Book Recommendation System.

Built for the full ~271K-book Kaggle Book-Crossing catalog. This module
deliberately never builds a dense book-vs-book cosine similarity matrix
(271,000 x 271,000 floats would need hundreds of GB of RAM). Instead it
keeps the TF-IDF matrix sparse and uses scikit-learn's NearestNeighbors
(cosine metric, sparse-aware brute-force search) to retrieve only the
top-N most similar rows for a single query book, on demand.
"""

import os
import pickle
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")
MATRIX_PATH = os.path.join(MODELS_DIR, "content_matrix.npz")
INDEX_PATH = os.path.join(MODELS_DIR, "content_index.pkl")


def train_content_model(books: pd.DataFrame):
    """Fit a TF-IDF vectorizer over the book catalog.

    Returns:
        vectorizer (TfidfVectorizer)
        tfidf_matrix (scipy.sparse.csr_matrix), shape (n_books, n_features)
        isbn_index (list[str]) -- row i of tfidf_matrix corresponds to isbn_index[i]
    """
    from utils.preprocess import build_content_features

    corpus = build_content_features(books)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    isbn_index = books["isbn"].reset_index(drop=True).tolist()

    return vectorizer, tfidf_matrix, isbn_index


def build_nn_index(tfidf_matrix) -> NearestNeighbors:
    """Build a cosine-similarity nearest-neighbor index over the sparse
    TF-IDF matrix. Fitting only stores a reference to the sparse matrix
    (no dense pairwise similarity is ever computed) -- the actual search
    happens lazily, per query, inside kneighbors()."""
    nn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    nn_model.fit(tfidf_matrix)
    return nn_model


def save_content_model(vectorizer, tfidf_matrix, isbn_index, models_dir: str = MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "tfidf.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    sp.save_npz(os.path.join(models_dir, "content_matrix.npz"), tfidf_matrix)
    with open(os.path.join(models_dir, "content_index.pkl"), "wb") as f:
        pickle.dump(isbn_index, f)


def load_content_model(models_dir: str = MODELS_DIR):
    with open(os.path.join(models_dir, "tfidf.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = sp.load_npz(os.path.join(models_dir, "content_matrix.npz"))
    with open(os.path.join(models_dir, "content_index.pkl"), "rb") as f:
        isbn_index = pickle.load(f)
    return vectorizer, tfidf_matrix, isbn_index


def content_model_artifacts_exist(models_dir: str = MODELS_DIR) -> bool:
    return (
        os.path.exists(os.path.join(models_dir, "tfidf.pkl"))
        and os.path.exists(os.path.join(models_dir, "content_matrix.npz"))
        and os.path.exists(os.path.join(models_dir, "content_index.pkl"))
    )


def get_content_recommendations(book_title: str, books: pd.DataFrame, nn_model,
                                 tfidf_matrix, isbn_index, top_n: int = 10) -> pd.DataFrame:
    """Return the top-N most similar books to the given book title using
    a sparse nearest-neighbor query against the single row for the
    selected book -- never a precomputed n x n similarity matrix."""
    cols = ["isbn", "title", "author", "publisher", "avg_rating", "rating_count"]
    empty = pd.DataFrame(columns=cols + ["similarity_score"])

    if not isbn_index or tfidf_matrix is None:
        return empty

    books_reset = books.reset_index(drop=True)
    matches = books_reset.index[books_reset["title"] == book_title].tolist()
    if not matches:
        return empty

    row_pos = matches[0]
    if row_pos >= tfidf_matrix.shape[0]:
        return empty

    n_neighbors = min(top_n + 1, tfidf_matrix.shape[0])
    distances, indices = nn_model.kneighbors(tfidf_matrix[row_pos], n_neighbors=n_neighbors)
    distances, indices = distances.flatten(), indices.flatten()

    pairs = [(i, 1 - d) for i, d in zip(indices, distances) if i != row_pos][:top_n]
    if not pairs:
        return empty

    result_isbns = [isbn_index[i] for i, _ in pairs]
    scores = {isbn_index[i]: round(float(score), 4) for i, score in pairs}

    recs = books_reset[books_reset["isbn"].isin(result_isbns)][cols].copy()
    recs["similarity_score"] = recs["isbn"].map(scores)
    recs = recs.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    return recs


if __name__ == "__main__":
    import sys
    sys.path.append(BASE_DIR)
    from utils.preprocess import load_and_clean_all

    books, ratings, users = load_and_clean_all()
    vectorizer, tfidf_matrix, isbn_index = train_content_model(books)
    save_content_model(vectorizer, tfidf_matrix, isbn_index)
    nn_model = build_nn_index(tfidf_matrix)
    print("Content-based model trained and saved.")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape} (sparse, no n x n matrix built)")

    sample_title = books.iloc[0]["title"]
    recs = get_content_recommendations(sample_title, books, nn_model, tfidf_matrix, isbn_index, top_n=5)
    print(f"\nTop 5 recommendations for '{sample_title}':")
    print(recs[["title", "author", "avg_rating"]])
