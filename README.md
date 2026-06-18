# Collaborative Filtering Recommender System

Movie rating prediction using memory-based collaborative filtering on the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset (100,000 ratings, 943 users, 1,682 movies).

## Methods

Five approaches are implemented and compared, from simple baselines to a tuned hybrid model:

| # | Method | Description |
|---|--------|-------------|
| 1 | User Average | Predict using the target user's mean rating |
| 2 | Item Average | Predict using the target item's mean rating |
| 3 | User-User KNN | k nearest users by Pearson similarity |
| 4 | Item-Item KNN | k nearest items by Pearson similarity |
| 5 | Hybrid KNN | Weighted combination of Methods 3 & 4 |

## Results

| Method | MAE | RMSE |
|--------|-----|------|
| User Average | 0.826 | 1.031 |
| Item Average | 0.855 | 1.072 |
| User-User KNN | 0.789 | 0.997 |
| Item-Item KNN | 0.770 | 0.990 |
| **Hybrid KNN** | **0.730** | **0.934** |

The hybrid model reduces MAE by ~11% over the best single-method baseline (Item-Item KNN).

## Key Design Decisions

**Significance weighting** — Pearson similarities computed from fewer than Δ co-rated items are down-weighted by `|co-rated| / Δ`. This reduces noise from user/item pairs with very little overlap.

**Hyperparameter tuning** — k (number of neighbours) and Δ (significance threshold) are selected via a held-out validation split from the training set. The test set is never used for tuning.

**Hybrid mixing** — The blending weight λ is also tuned on validation. The optimal λ ≈ 0.45 (slightly favouring item-based predictions), which reflects that item-item similarities are generally more stable on this dataset.

## How to Run

**Requirements:** Python 3.7+, NumPy, Pandas, scikit-learn, Matplotlib

```bash
# Download the dataset
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip

# Install dependencies
pip install numpy pandas scikit-learn matplotlib

# Open the notebook
jupyter notebook collaborative_filtering_recommender.ipynb
```

> Note: Methods 3–5 compute full similarity matrices (943×943 and 1682×1682) which is computationally expensive. Expect ~15–30 minutes runtime without optimisation.

## Dataset

GroupLens Research. (1998). *MovieLens 100K Dataset*. University of Minnesota.  
https://grouplens.org/datasets/movielens/100k/