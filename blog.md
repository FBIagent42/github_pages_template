# Predicting Board Game Type with KNN: A Beginner's Guide

## Introduction

Ever wondered if you could predict the **type of a board game**—like *strategy*, *card*, or *deception*—just by looking at its attributes? In this tutorial, we’ll walk through how to use the **K-Nearest Neighbors (KNN)** algorithm to classify board games based on features like player count, play time, and complexity.

This guide is perfect for Stat-386 students who want to apply machine learning to real-world data in a reproducible and interpretable way.

---

## What You'll Learn

- How to prepare board game data for modeling
- How KNN works and why it's useful for classification
- How to implement KNN in Python using `scikit-learn`
- How to evaluate model performance
- How to interpret predictions

---

## Step 1: Understanding the Data

We’ll use a simplified dataset of board games with the following attributes:

| Name             | Min Players | Max Players | Play Time | Complexity | Type        |
|------------------|-------------|-------------|-----------|------------|-------------|
| Codenames        | 2           | 8           | 15        | 1.2        | Party       |
| Dominion         | 2           | 4           | 30        | 2.3        | Card Game   |
| Twilight Struggle| 2           | 2           | 180       | 3.8        | Strategy    |

> **Note**: Complexity is a rating from 1 (easy) to 5 (complex), sourced from [BoardGameGeek](https://boardgamehat is KNN?

K-Nearest Neighbors is a **non-parametric**, **instance-based** learning algorithm. It classifies a new data point based on the majority label of its *k* closest neighbors in the feature space.

### The KNN Formula

To compute the distance between two games, we use **Euclidean distance**:

$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$
Where:
- \( x \) and \( y \) are feature vectors
- \( n \) is the number of features

---

## Step 3: Implementing KNN in Python

Here’s how to build a simple KNN classifier using `scikit-learn`.

```python
# Import libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
data = {
    'min_players': [2, 2, 2],
    'max_players': [8, 4, 2],
    'play_time': [15, 30, 180],
    'complexity': [1.2, 2.3, 3.8],
    'type': ['Party', 'Card Game', 'Strategy']
}

df = pd.DataFrame(data)

# Encode labels
df['type_encoded'] = df['type'].astype('category').cat.codes

# Features and target
X = df[['min_players', 'max_players', 'play_time', 'complexity']]
y = df['type_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# KNN model
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


