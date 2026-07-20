# dataset/

This folder is intentionally empty in the repository.

Download the **Book Recommendation Dataset** from Kaggle:
https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

Unzip it and copy exactly these three files into this folder, **without
renaming them**:

```
dataset/
├── Books.csv
├── Ratings.csv
└── Users.csv
```

`utils/preprocess.py` reads these exact filenames (case-sensitive) and
handles encoding issues, missing values, duplicates, invalid years, and
missing images automatically. No other setup is required — once the
three CSVs are here, run:

```bash
python train_models.py
streamlit run app.py
```

(`streamlit run app.py` will also auto-train on first launch if you
skip `train_models.py`, but running it first is faster for a 271K-book
catalog.)
