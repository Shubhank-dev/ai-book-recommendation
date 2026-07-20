# 📚 AI Book Recommendation System

An end-to-end, production-quality **Machine Learning powered book recommendation platform**, built on the real-world **[Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)** from Kaggle (278K users, 271K books, 1.1M ratings collected from the Book-Crossing community). The app combines **Content-Based Filtering** (TF-IDF + sparse Nearest-Neighbor cosine similarity) and **Collaborative Filtering** (Surprise SVD) inside a modern, responsive **Streamlit** interface — no Flask or Django required.

---

## 1. Introduction

Online bookstores carry catalogs far too large for any single reader to browse manually. This project solves that discovery problem with two complementary ML-driven recommendation strategies:

- **"More like this"** — find books similar in title, author, and publisher to a book the reader already loves (Content-Based Filtering).
- **"Picked for you"** — predict how a specific reader would rate books they haven't read yet, based on the collective rating patterns of every user in the system (Collaborative Filtering).

The system also surfaces a **Trending Books** dashboard and a global **Search** experience, all wrapped in a polished, dark-themed, card-based UI.

---

## 2. Features

- 🏠 **Home** — hero section, live dataset metrics, and featured books.
- 🔎 **Content-Based Recommendation** — searchable book dropdown → Top 10 similar books via TF-IDF + Nearest-Neighbor cosine similarity, showing title, author, publisher, and average rating.
- 👤 **Personalized Recommendation** — enter a User ID → Top 10 personalized picks via a trained Surprise SVD model, with a graceful cold-start fallback for unknown users.
- 🔥 **Trending Books** — Top Rated, Most Popular, Most Reviewed, and Decade Leaders tabs.
- 🔍 **Search** — search by title or author from the sidebar (available on every page) or the dedicated Trending page.
- 🎨 **Professional Streamlit UI** — wide layout, sidebar navigation, custom CSS styling, metric cards, book cards, loading spinners, success/error messages, and a fully responsive grid layout.
- 🧹 **Robust preprocessing** — automatic encoding detection, missing values, duplicate books, duplicate ratings, invalid publication years, missing authors/publishers, and missing cover images all handled, with everything merged on ISBN.
- ⚡ **Memory-efficient at full scale** — no dense book-vs-book similarity matrix is ever built (271,000² floats would need hundreds of GB of RAM); a sparse TF-IDF matrix + `NearestNeighbors` handles content similarity, and personalized recommendations score a bounded candidate pool instead of the full catalog.
- 💾 **Persisted ML models** — TF-IDF vectorizer + sparse matrix + ISBN index, and the SVD model, are trained once and saved to disk, then loaded instantly by the app (with automatic on-the-fly retraining if the artifacts are ever missing).

---

## 3. Folder Structure

```
AI-Book-Recommendation-System/
├── app.py                      # Streamlit application (entry point)
├── train_models.py             # Trains and saves all ML models
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── dataset/
│   └── README.md                # Empty — copy Books.csv, Ratings.csv, Users.csv here
├── models/                      # Trained artifacts land here (empty until you train)
├── notebooks/
│   └── training.ipynb          # Full EDA + training walkthrough
├── utils/
│   ├── preprocess.py           # Data loading & cleaning (real Kaggle schema)
│   ├── content_based.py        # TF-IDF + sparse Nearest-Neighbor engine
│   ├── collaborative.py        # Surprise SVD engine
│   └── helper.py                # Trending / search / formatting helpers
└── assets/
    ├── logo.png
    └── screenshots/
```

---

## 4. Installation

### Requirements
- Python 3.12+
- pip

### Steps

```bash
# 1. Unzip / clone the project, then move into it
cd AI-Book-Recommendation-System

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset (see section 5) and place the 3 CSVs in dataset/

# 5. Train the ML models (creates the model artifacts inside models/)
python train_models.py

# 6. Launch the app
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.

> `app.py` will also train and save the models automatically on first run if `models/` is empty — so you can skip step 5 and just run `streamlit run app.py` directly once the dataset is in place. Running `train_models.py` first is recommended for the full 271K-book catalog, since training takes a little while and you'll want to see progress logged in the terminal rather than a Streamlit spinner.

---

## 5. Dataset

This project runs on the real-world **[Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)** on Kaggle.

**This repository does not include the dataset and does not download it automatically.** Download it yourself:

1. Go to https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset
2. Download and unzip it
3. Copy the three files into `dataset/`, **without renaming them**:

```
dataset/
├── Books.csv
├── Ratings.csv
└── Users.csv
```

| File | Source columns | Notes |
|---|---|---|
| `Books.csv` | `ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M, Image-URL-L` | Real titles, authors, years, publishers, and cover images |
| `Ratings.csv` | `User-ID, ISBN, Book-Rating` | Real explicit ratings on Book-Crossing's native 1–10 scale (`0` = no rating given / implicit interaction) |
| `Users.csv` | `User-ID, Location, Age` | Real (self-reported) location/age from Book-Crossing members |

`utils/preprocess.py` reads these exact filenames and handles the rest automatically:

- **Encoding issues** — tries UTF-8, then Latin-1, then CP1252, and skips malformed rows, since the raw Book-Crossing export mixes encodings.
- **Missing values** — missing authors/publishers are filled with clear placeholders (`"Unknown Author"` / `"Unknown Publisher"`); rows with no ISBN/title/author at all are dropped.
- **Duplicate books** — exact duplicate rows and duplicate ISBNs are removed (ISBN is the merge key, so it must be unique).
- **Duplicate ratings** — a user rating the same book more than once keeps only the latest rating.
- **Invalid publication years** — years outside a sane range (before 1450 or more than a year in the future) are treated as unknown rather than dropped or guessed.
- **Missing image URLs** — `Image-URL-M` is used first, falling back to `Image-URL-L`, then `Image-URL-S`, then a neutral "no cover" placeholder image.
- **Implicit ratings** — `Book-Rating == 0` (no explicit rating given) is excluded from both the average-rating stats and the collaborative-filtering training data.
- **Merging** — books, ratings, and users are joined on `ISBN` / `User-ID` throughout, and only ratings that reference a valid (cleaned) book and user are kept.

After placing the dataset, run `python train_models.py` to train the models on it. No further code changes are needed.

---

## 6. Machine Learning Models

### Content-Based Filtering
- **Feature engineering**: `title (x2) + author (x2) + publisher` combined into one text field per book. (The real dataset has no genre or description column, so the model is built only from real, non-fabricated fields.)
- **Vectorization**: `sklearn.feature_extraction.text.TfidfVectorizer` (English stop-words removed, 1–2 n-grams, up to 50,000 features).
- **Similarity search**: `sklearn.neighbors.NearestNeighbors` (cosine metric) queried against the sparse TF-IDF matrix. **No dense book-vs-book similarity matrix is ever built** — at 271,000 books, a 271,000 × 271,000 dense matrix would need hundreds of gigabytes of RAM. Instead, each query computes similarity against the sparse matrix for a single book on demand and returns only the top-N neighbors.
- **Persistence**: `models/tfidf.pkl` (vectorizer), `models/content_matrix.npz` (sparse TF-IDF matrix), `models/content_index.pkl` (row → ISBN lookup).

### Collaborative Filtering
- **Library**: [Surprise](http://surpriselib.com/) (`scikit-surprise`).
- **Algorithm**: `SVD` (Singular Value Decomposition / matrix factorization), `n_factors=50`, `n_epochs=20`, on the native 1–10 rating scale.
- **Training**: an 80/20 train/test split is used to report RMSE, then the model is refit on the **full** dataset for maximum coverage before being saved.
- **Inference**: for a given user, predictions are computed only for a bounded candidate pool (the most-rated unrated books, up to a few thousand) rather than the entire 271K-book catalog, so recommendations stay responsive regardless of catalog size. The Top 10 highest predicted ratings are returned.
- **Cold-start handling**: users with no rating history fall back to the Trending "Top Rated" list.
- **Persistence**: `models/svd_model.pkl`.

### Notebook
`notebooks/training.ipynb` contains the full walkthrough: raw data loading → cleaning → EDA → TF-IDF + Nearest-Neighbor training → SVD training & RMSE evaluation → saving all model artifacts.

---

## 7. Screenshots

> Screenshots are saved inside `assets/screenshots/`. Run the app locally with `streamlit run app.py` and capture the Home, Content-Based, Personalized, and Trending pages to populate this folder — placeholders are provided below.

| Page | File |
|---|---|
| Home | `assets/screenshots/home.png` |
| Content-Based Recommendation | `assets/screenshots/content_based.png` |
| Personalized Recommendation | `assets/screenshots/personalized.png` |
| Trending Books | `assets/screenshots/trending.png` |

---

## 8. Requirements

See [`requirements.txt`](requirements.txt) for the full pinned dependency list. Core libraries:

- `streamlit` — UI framework
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — TF-IDF vectorization, Nearest-Neighbor similarity search
- `scikit-surprise` — SVD collaborative filtering
- `scipy`, `joblib` — sparse matrix support / model persistence support

---

## 9. Future Improvements

- Hybrid recommender that blends content-based and collaborative scores (weighted ensemble).
- Real-time model retraining triggered from the UI when new ratings are submitted.
- User authentication and a persistent "My Ratings" page so users can rate books directly in-app.
- Approximate nearest-neighbor indexing (e.g. FAISS/Annoy) for sub-linear content-similarity search at even larger scale.
- Deploy to Streamlit Community Cloud / Docker with a CI pipeline that retrains models on a schedule.
- Add explainability ("recommended because you liked X") to both recommendation types.
- A/B testing framework to compare recommendation strategies on click-through rate.

---

## 10. License

This project is released under the [MIT License](LICENSE).
