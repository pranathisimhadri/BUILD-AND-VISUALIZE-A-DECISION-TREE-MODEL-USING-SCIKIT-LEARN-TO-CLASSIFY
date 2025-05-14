# BUILD-AND-VISUALIZE-A-DECISION-TREE-MODEL-USING-SCIKIT-LEARN-TO-CLASSIFY  OR PREDICT OUTCOMES ON A CHOSEN DATASET
COMPANY : CODTECH IT SOLUTIONS

NAME :Pranathi Simhadri

INTERN ID : CT04DM549

DOMAIN : MACHINE LEARNING

DURATION : 4 WEEEKS

MENTOR : NEELA SANTOSH

## 🎯 Project: Movie Recommendation via Matrix Factorization in PyTorch

This project builds a **Collaborative Filtering-based Movie Recommender System** using **Matrix Factorization** with **PyTorch**, optimized for **GPU acceleration (CUDA)**.

It learns **latent embeddings** for users and movies such that their dot product approximates user ratings. This enables predicting user preferences and recommending top-N movies.

### 🔍 Objective

Use matrix factorization to decompose the user-item rating matrix into two low-dimensional matrices: one for user features and one for movie features. This method captures hidden patterns in user behavior and item popularity.

---

## 🧠 Technical Summary

### 1. **Preprocessing**

* Load ratings (`ratings.csv`) via pandas.
* Convert user/movie IDs to integer indices using `factorize()`.
* Split data into training/test sets.

### 2. **Dataset & Loader**

* Custom `RatingsDataset` returns user, movie, and rating.
* `DataLoader` enables mini-batch training.

### 3. **Model**

* `user_embedding` and `movie_embedding` via `nn.Embedding`.
* Dot product predicts the rating.
* Loss: Mean Squared Error (MSE).

### 4. Training

* Runs on GPU if available via `torch.device("cuda")`.
* Efficient training for large datasets.

### 5. Saving & Predicting

* Save/load model with `torch.save()` and `torch.load()`.
* Predict ratings or generate the full rating matrix.

---

## ⚙️ Features

#✅ Matrix factorization with collaborative filtering
# ⚡ CUDA support for faster training
# 💾 Save/load model checkpoints
# 📈 Predict individual ratings or full matrix
# 🧱 Easy to extend (e.g., top-N recs, RMSE, implicit feedback)

---

## 📦 Ideal For

#Learning recommender systems in PyTorch
# Personal/movie recommendation tools
# Academic/research use cases
# Prototyping scalable recommenders


#OUTPU

![Image](https://github.com/user-attachments/assets/cfde1bae-a006-4aa1-b740-562ad86a555a)

