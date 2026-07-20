"""
train_models.py
----------------
Master training script for the AI Book Recommendation System.

Runs the full pipeline:
    1. Load and clean the real Kaggle dataset (Books.csv, Ratings.csv, Users.csv)
    2. Train the Content-Based model (TF-IDF + sparse Nearest-Neighbor index)
    3. Train the Collaborative Filtering model (Surprise SVD)
    4. Persist all model artifacts to the models/ directory

Run this once before launching the Streamlit app:
    python train_models.py

Requires dataset/Books.csv, dataset/Ratings.csv, and dataset/Users.csv
(downloaded manually from
https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocess import load_and_clean_all
from utils.content_based import train_content_model, save_content_model
from utils.collaborative import train_svd_model, save_svd_model


def main():
    start = time.time()
    print("=" * 60)
    print("AI Book Recommendation System - Model Training Pipeline")
    print("=" * 60)

    print("\n[1/3] Loading and cleaning dataset...")
    books, ratings, users = load_and_clean_all()
    print(f"    Books:   {len(books):,}")
    print(f"    Ratings: {len(ratings):,}")
    print(f"    Users:   {len(users):,}")

    print("\n[2/3] Training Content-Based model (TF-IDF + sparse Nearest Neighbors)...")
    vectorizer, tfidf_matrix, isbn_index = train_content_model(books)
    save_content_model(vectorizer, tfidf_matrix, isbn_index)
    print(f"    TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"    TF-IDF matrix shape:    {tfidf_matrix.shape} (sparse -- no n x n matrix built)")
    print("    Saved -> models/tfidf.pkl, models/content_matrix.npz, models/content_index.pkl")

    print("\n[3/3] Training Collaborative Filtering model (Surprise SVD)...")
    svd_model, rmse = train_svd_model(ratings)
    save_svd_model(svd_model)
    print(f"    Test RMSE: {rmse:.4f}")
    print("    Saved -> models/svd_model.pkl")

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"All models trained successfully in {elapsed:.2f} seconds.")
    print("You can now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
