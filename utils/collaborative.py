"""
collaborative.py
-----------------
Collaborative Filtering engine for the AI Book Recommendation System.
Uses the Surprise library's SVD (Singular Value Decomposition) algorithm
to learn latent factors from the user-item (user_id, isbn, rating)
matrix and generate personalized recommendations.
"""

import os
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SVD_MODEL_PATH = os.path.join(MODELS_DIR, "svd_model.pkl")

# Book-Crossing explicit ratings are on a 1-10 scale
RATING_SCALE = (1, 10)

# When generating personalized recommendations we do NOT score every
# unrated book in a ~271K-book catalog (that would mean 271K individual
# model.predict() calls per request). Instead we restrict scoring to
# this many of the most-rated unseen books -- the SVD model has the
# most reliable signal for those anyway -- which keeps the page
# responsive regardless of catalog size.
CANDIDATE_POOL_SIZE = 4000


def train_svd_model(ratings: pd.DataFrame, n_factors: int = 50,
                     n_epochs: int = 20, test_size: float = 0.2,
                     random_state: int = 42):
    """Train an SVD collaborative filtering model on the ratings dataframe.

    Expects columns: user_id, isbn, rating

    Returns:
        model (SVD): the trained model, refit on the full dataset
        rmse (float): root mean squared error on the held-out test split
    """
    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(ratings[["user_id", "isbn", "rating"]], reader)

    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)

    model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)

    # Refit on the FULL dataset for maximum coverage in production
    full_trainset = data.build_full_trainset()
    model.fit(full_trainset)

    return model, rmse


def save_svd_model(model, models_dir: str = MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "svd_model.pkl"), "wb") as f:
        pickle.dump(model, f)


def load_svd_model(models_dir: str = MODELS_DIR):
    with open(os.path.join(models_dir, "svd_model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model


def get_personalized_recommendations(user_id: int, model, books: pd.DataFrame,
                                      ratings: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Predict ratings for a bounded candidate pool of unrated books and
    return the top-N with the highest predicted rating.

    To stay responsive on the full ~271K-book catalog, predictions are
    only computed for the CANDIDATE_POOL_SIZE most-rated books the user
    hasn't already rated, instead of scoring the entire catalog.
    """
    rated_isbns = set(ratings.loc[ratings["user_id"] == user_id, "isbn"].unique())

    candidates = books[~books["isbn"].isin(rated_isbns)]
    pool_size = max(CANDIDATE_POOL_SIZE, top_n * 20)
    candidates = candidates.sort_values("rating_count", ascending=False).head(pool_size)

    if candidates.empty:
        candidates = books[~books["isbn"].isin(rated_isbns)]

    predictions = []
    for isbn in candidates["isbn"]:
        pred = model.predict(uid=user_id, iid=isbn)
        predictions.append((isbn, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]

    isbns = [b for b, _ in top_predictions]
    scores = {b: round(s, 2) for b, s in top_predictions}

    recs = books[books["isbn"].isin(isbns)][
        ["isbn", "title", "author", "publisher", "avg_rating", "rating_count"]
    ].copy()
    recs["predicted_rating"] = recs["isbn"].map(scores)
    recs = recs.sort_values("predicted_rating", ascending=False).reset_index(drop=True)

    return recs


def is_known_user(user_id: int, ratings: pd.DataFrame) -> bool:
    """Check whether a user exists in the training ratings data
    (cold-start users get a fallback in app.py)."""
    return user_id in set(ratings["user_id"].unique())


if __name__ == "__main__":
    import sys
    sys.path.append(BASE_DIR)
    from utils.preprocess import load_and_clean_all

    books, ratings, users = load_and_clean_all()
    model, rmse = train_svd_model(ratings)
    save_svd_model(model)
    print(f"SVD model trained. Test RMSE: {rmse:.4f}")

    sample_user = int(ratings["user_id"].iloc[0])
    recs = get_personalized_recommendations(sample_user, model, books, ratings, top_n=5)
    print(f"\nTop 5 personalized recommendations for user {sample_user}:")
    print(recs[["title", "author", "predicted_rating"]])
