"""
app.py
------
AI Book Recommendation System — Streamlit Application

A professional, ML-powered book recommendation platform, built on the
real-world Kaggle Book Recommendation Dataset (arashnic/book-recommendation-dataset),
featuring:
    - Content-Based Filtering (TF-IDF + sparse Nearest-Neighbor search)
    - Collaborative Filtering (Surprise SVD)
    - Trending Books dashboard
    - Search by title / author

Run with:
    streamlit run app.py
"""

import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocess import load_and_clean_all
from utils.content_based import (
    load_content_model,
    get_content_recommendations,
    train_content_model,
    save_content_model,
    build_nn_index,
    content_model_artifacts_exist,
)
from utils.collaborative import (
    load_svd_model,
    get_personalized_recommendations,
    is_known_user,
    train_svd_model,
    save_svd_model,
)
from utils.helper import (
    get_top_rated_books,
    get_most_popular_books,
    get_most_reviewed_books,
    get_highest_rated_decade_leaders,
    search_books,
    format_rating_stars,
    truncate_text,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom CSS — professional, modern styling
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .main {
        background-color: #0f1116;
    }
    .stApp {
        background: linear-gradient(180deg, #0f1116 0%, #14171f 100%);
    }
    h1, h2, h3, h4 {
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        color: #f5f6fa;
    }
    p, span, label, div {
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #b8bcc8;
        margin-bottom: 1.5rem;
    }
    .book-card {
        background: #1a1d29;
        border: 1px solid #2a2e3d;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
        transition: all 0.2s ease-in-out;
    }
    .book-card:hover {
        border-color: #6C63FF;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(108, 99, 255, 0.15);
    }
    .book-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f5f6fa;
        margin-bottom: 0.15rem;
    }
    .book-meta {
        font-size: 0.88rem;
        color: #9298a8;
        margin-bottom: 0.4rem;
    }
    .book-desc {
        font-size: 0.85rem;
        color: #b8bcc8;
        line-height: 1.4;
    }
    .rating-badge {
        display: inline-block;
        background: rgba(255, 193, 7, 0.15);
        color: #ffc107;
        border-radius: 8px;
        padding: 0.15rem 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .genre-badge {
        display: inline-block;
        background: rgba(108, 99, 255, 0.15);
        color: #a29bfe;
        border-radius: 8px;
        padding: 0.15rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .score-badge {
        display: inline-block;
        background: rgba(255, 101, 132, 0.15);
        color: #ff6584;
        border-radius: 8px;
        padding: 0.15rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.4rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #14171f;
        border-right: 1px solid #2a2e3d;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6C63FF, #a29bfe);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 18px rgba(108, 99, 255, 0.35);
        transform: translateY(-1px);
    }
    div[data-testid="stMetric"] {
        background: #1a1d29;
        border: 1px solid #2a2e3d;
        border-radius: 12px;
        padding: 0.8rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached data / model loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    books, ratings, users = load_and_clean_all()
    return books, ratings, users


@st.cache_resource(show_spinner=False)
def get_content_model(_books):
    """Load the pretrained content-based model, training it on the fly
    if the pickle files are missing (first-run convenience). Returns a
    fitted NearestNeighbors index (never a dense n x n similarity
    matrix) plus the sparse TF-IDF matrix and its ISBN row index."""
    if content_model_artifacts_exist(MODELS_DIR):
        vectorizer, tfidf_matrix, isbn_index = load_content_model()
    else:
        vectorizer, tfidf_matrix, isbn_index = train_content_model(_books)
        save_content_model(vectorizer, tfidf_matrix, isbn_index)
    nn_model = build_nn_index(tfidf_matrix)
    return nn_model, tfidf_matrix, isbn_index


@st.cache_resource(show_spinner=False)
def get_collab_model(_ratings):
    """Load the pretrained SVD model, training it on the fly if the
    pickle file is missing (first-run convenience)."""
    svd_path = os.path.join(MODELS_DIR, "svd_model.pkl")
    if os.path.exists(svd_path):
        model = load_svd_model()
    else:
        model, _ = train_svd_model(_ratings)
        save_svd_model(model)
    return model


# --------------------------------------------------------------------------
# Reusable UI components
# --------------------------------------------------------------------------
def _year_display(year_val) -> str:
    """Safely format a nullable publication year for display."""
    if pd.isna(year_val):
        return ""
    try:
        return str(int(year_val))
    except (TypeError, ValueError):
        return ""


def render_book_card(row: dict, badge_label: str = None, badge_value: str = None):
    stars = format_rating_stars(row.get("avg_rating", 0), max_rating=10.0)
    desc = truncate_text(row.get("description", ""), 150) if row.get("description") else ""
    year_str = _year_display(row.get("year"))

    extra_badge = ""
    if badge_label and badge_value is not None:
        extra_badge = f'<span class="score-badge">{badge_label}: {badge_value}</span>'

    st.markdown(
        f"""
        <div class="book-card">
            <div class="book-title">{row.get('title', 'Untitled')}</div>
            <div class="book-meta">by {row.get('author', 'Unknown Author')} · {year_str}</div>
            <span class="rating-badge">⭐ {stars} ({row.get('avg_rating', 0)}/10)</span>
            <span class="genre-badge">{row.get('publisher', 'Unknown Publisher')}</span>
            {extra_badge}
            <div class="book-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_book_grid(df: pd.DataFrame, columns: int = 2, badge_col: str = None, badge_label: str = None):
    rows = df.to_dict(orient="records")
    for i in range(0, len(rows), columns):
        cols = st.columns(columns)
        for j, col in enumerate(cols):
            if i + j < len(rows):
                row = rows[i + j]
                badge_value = row.get(badge_col) if badge_col else None
                with col:
                    render_book_card(row, badge_label=badge_label, badge_value=badge_value)


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.markdown("## 📚 AI Book Recommender")
st.sidebar.markdown("---")

PAGES = [
    "🏠 Home",
    "🔎 Content Based Recommendation",
    "👤 Personalized Recommendation",
    "🔥 Trending Books",
    "ℹ️ About",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Quick Search")
sidebar_query = st.sidebar.text_input("Search by title or author", key="sidebar_search")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python, Streamlit, Scikit-learn & Surprise")

# --------------------------------------------------------------------------
# Load data (with spinner + error handling)
# --------------------------------------------------------------------------
try:
    with st.spinner("Loading and preparing dataset..."):
        books, ratings, users = get_data()
except FileNotFoundError as e:
    st.error(
        "Dataset files not found. Please download the dataset from "
        "[Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) "
        "and place `Books.csv`, `Ratings.csv`, and `Users.csv` (exact filenames) "
        "inside the `dataset/` folder.\n\n"
        f"Details: {e}"
    )
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the dataset: {e}")
    st.stop()

# Sidebar quick search results (shown on every page if a query is typed)
if sidebar_query:
    st.sidebar.markdown("#### Results")
    quick_results = search_books(books, sidebar_query).head(5)
    if quick_results.empty:
        st.sidebar.info("No matches found.")
    else:
        for _, r in quick_results.iterrows():
            st.sidebar.write(f"**{r['title']}** — {r['author']}")


# ==========================================================================
# PAGE: HOME
# ==========================================================================
if page == "🏠 Home":
    st.markdown('<div class="hero-title">AI Book Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">An intelligent, ML-powered book recommendation platform '
        'built on the real-world Book-Crossing dataset — combining Content-Based Filtering and '
        'Collaborative Filtering to help readers discover their next favorite book.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📚 Total Books", f"{len(books):,}")
    col2.metric("👥 Total Users", f"{len(users):,}")
    col3.metric("⭐ Total Ratings", f"{len(ratings):,}")
    col4.metric("🏢 Publishers", f"{books['publisher'].nunique():,}")

    st.markdown("### ")
    st.markdown("### 🚀 What can you do here?")

    feat_col1, feat_col2, feat_col3 = st.columns(3)
    with feat_col1:
        st.markdown(
            """
            <div class="book-card">
                <div class="book-title">🔎 Content-Based Recommendation</div>
                <div class="book-desc">Pick a book you love and instantly discover
                similar titles using TF-IDF and Nearest-Neighbor cosine similarity
                across title, author and publisher.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with feat_col2:
        st.markdown(
            """
            <div class="book-card">
                <div class="book-title">👤 Personalized Recommendation</div>
                <div class="book-desc">Enter your User ID and get a
                personalized Top 10 list generated by a Collaborative
                Filtering model (Surprise SVD) trained on real rating patterns.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with feat_col3:
        st.markdown(
            """
            <div class="book-card">
                <div class="book-title">🔥 Trending Books</div>
                <div class="book-desc">Explore the highest rated and most
                reviewed books in the catalog, updated directly from the
                cleaned rating data.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### ")
    st.markdown("### 🌟 Featured Books")
    featured = get_top_rated_books(books, min_votes=3, top_n=4)
    render_book_grid(featured, columns=4)

    st.success("Use the sidebar to navigate between pages and explore recommendations!")


# ==========================================================================
# PAGE: CONTENT BASED RECOMMENDATION
# ==========================================================================
elif page == "🔎 Content Based Recommendation":
    st.markdown("## 🔎 Content-Based Recommendation")
    st.markdown(
        "Select a book you enjoyed, and our TF-IDF + Nearest-Neighbor cosine "
        "similarity engine will find the **Top 10** most similar books based on "
        "title, author, and publisher."
    )

    book_titles = sorted(books["title"].unique().tolist())
    selected_title = st.selectbox(
        "📖 Search and select a book",
        options=book_titles,
        index=0 if book_titles else None,
        help="Start typing to filter the dropdown list.",
    )

    if selected_title:
        selected_row = books[books["title"] == selected_title].iloc[0]
        st.markdown("#### You selected:")
        render_book_card(selected_row.to_dict())

    find_btn = st.button("🎯 Find Similar Books", use_container_width=False)

    if find_btn:
        try:
            with st.spinner("Analyzing book content and computing similarity scores..."):
                nn_model, tfidf_matrix, isbn_index = get_content_model(books)
                recommendations = get_content_recommendations(
                    selected_title, books, nn_model, tfidf_matrix, isbn_index, top_n=10
                )

            if recommendations.empty:
                st.warning("No similar books could be found for this title.")
            else:
                st.success(f"Found {len(recommendations)} books similar to **{selected_title}**!")
                st.markdown("### 📚 Top 10 Recommendations")
                render_book_grid(
                    recommendations, columns=2,
                    badge_col="similarity_score", badge_label="Match"
                )

                with st.expander("📊 View as table"):
                    st.dataframe(
                        recommendations[
                            ["title", "author", "publisher", "avg_rating", "similarity_score"]
                        ].rename(columns={
                            "title": "Title", "author": "Author", "publisher": "Publisher",
                            "avg_rating": "Avg Rating", "similarity_score": "Similarity"
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
        except Exception as e:
            st.error(f"An error occurred while generating recommendations: {e}")


# ==========================================================================
# PAGE: PERSONALIZED RECOMMENDATION
# ==========================================================================
elif page == "👤 Personalized Recommendation":
    st.markdown("## 👤 Personalized Recommendation")
    st.markdown(
        "Enter your **User ID** below to receive personalized book "
        "recommendations powered by a Collaborative Filtering model "
        "(Surprise SVD) trained on the full user-item rating matrix."
    )

    min_uid = int(users["user_id"].min())
    max_uid = int(users["user_id"].max())

    input_col, btn_col = st.columns([3, 1])
    with input_col:
        user_id_input = st.number_input(
            f"🔢 Enter your User ID (available range: {min_uid} - {max_uid})",
            min_value=1,
            step=1,
            value=min_uid,
        )
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("✨ Get Recommendations", use_container_width=True)

    if generate_btn:
        user_id = int(user_id_input)

        if not is_known_user(user_id, ratings):
            st.warning(
                f"User ID **{user_id}** has no rating history in our dataset "
                "(cold-start user). Showing top-rated books instead as a fallback."
            )
            fallback = get_top_rated_books(books, min_votes=5, top_n=10)
            render_book_grid(fallback, columns=2)
        else:
            try:
                with st.spinner("Running SVD model to compute your personalized recommendations..."):
                    svd_model = get_collab_model(ratings)
                    recommendations = get_personalized_recommendations(
                        user_id, svd_model, books, ratings, top_n=10
                    )

                if recommendations.empty:
                    st.warning("No recommendations could be generated for this user.")
                else:
                    st.success(f"Here are your Top 10 personalized picks, User {user_id}!")
                    render_book_grid(
                        recommendations, columns=2,
                        badge_col="predicted_rating", badge_label="Predicted"
                    )

                    with st.expander("📊 View as table"):
                        st.dataframe(
                            recommendations[
                                ["title", "author", "publisher", "avg_rating", "predicted_rating"]
                            ].rename(columns={
                                "title": "Title", "author": "Author", "publisher": "Publisher",
                                "avg_rating": "Avg Rating", "predicted_rating": "Predicted Rating"
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )

                    with st.expander("📈 Your rating history"):
                        user_history = ratings[ratings["user_id"] == user_id].merge(
                            books[["isbn", "title", "author", "publisher"]], on="isbn"
                        )[["title", "author", "publisher", "rating"]]
                        st.dataframe(user_history, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"An error occurred while generating recommendations: {e}")


# ==========================================================================
# PAGE: TRENDING BOOKS
# ==========================================================================
elif page == "🔥 Trending Books":
    st.markdown("## 🔥 Trending Books")
    st.markdown("Discover what readers are loving right now across the catalog.")

    search_col, _ = st.columns([2, 3])
    with search_col:
        trend_query = st.text_input("🔍 Search by title or author", key="trending_search")

    if trend_query:
        results = search_books(books, trend_query)
        st.markdown(f"### Search results for \"{trend_query}\" ({len(results)} found)")
        if results.empty:
            st.info("No books matched your search. Try a different title or author.")
        else:
            render_book_grid(results.head(20), columns=2)
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["⭐ Top Rated", "🔥 Most Popular", "📝 Most Reviewed", "📅 Decade Leaders"]
        )

        with tab1:
            st.markdown("#### Highest Average Rating (min. 5 ratings)")
            top_rated = get_top_rated_books(books, min_votes=5, top_n=10)
            render_book_grid(top_rated, columns=2, badge_col="avg_rating", badge_label="Rating")

        with tab2:
            st.markdown("#### Most Popular Books")
            popular = get_most_popular_books(books, top_n=10)
            render_book_grid(popular, columns=2, badge_col="rating_count", badge_label="Reviews")

        with tab3:
            st.markdown("#### Most Reviewed Books")
            reviewed = get_most_reviewed_books(books, top_n=10)
            render_book_grid(reviewed, columns=2, badge_col="rating_count", badge_label="Reviews")

        with tab4:
            st.markdown("#### Top Book by Publication Decade")
            leaders = get_highest_rated_decade_leaders(books, top_n=10)
            render_book_grid(leaders, columns=2, badge_col="avg_rating", badge_label="Rating")


# ==========================================================================
# PAGE: ABOUT
# ==========================================================================
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About This Project")

    st.markdown(
        """
        <div class="book-card">
            <div class="book-title">📚 AI Book Recommendation System</div>
            <div class="book-desc">
            This project is an end-to-end Machine Learning application built on the
            real-world Kaggle Book Recommendation Dataset, designed to help readers
            discover books they'll love using two complementary recommendation strategies.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧠 Machine Learning Approach")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Content-Based Filtering**
            - TF-IDF Vectorization on title, author & publisher
            - Sparse Nearest-Neighbor cosine similarity (no dense n x n matrix)
            - Great for "more like this" recommendations
            - Works even for books with few or no ratings
            """
        )
    with col2:
        st.markdown(
            """
            **Collaborative Filtering**
            - Surprise SVD (Singular Value Decomposition)
            - Learns latent user & book factors from the rating matrix
            - Predicts how a user would rate unseen books
            - Great for personalized "for you" recommendations
            """
        )

    st.markdown("### 🛠️ Tech Stack")
    st.write(
        "Python 3.12+, Streamlit, Pandas, NumPy, Scikit-learn, "
        "Surprise, TF-IDF, Nearest Neighbors, Pickle"
    )

    st.markdown("### 📦 Dataset")
    st.write(
        "Real-world [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) "
        "from Kaggle — Books.csv, Ratings.csv, and Users.csv, merged on ISBN."
    )

    st.markdown("### 👨‍💻 Project Goals")
    st.write(
        "Demonstrate a production-quality, fully runnable recommendation "
        "system architecture — from raw, messy real-world data through preprocessing, "
        "memory-efficient model training, and an interactive Streamlit front end — without "
        "relying on any web framework such as Flask or Django."
    )

    st.info("Made with ❤️ using Streamlit and Scikit-learn.")
