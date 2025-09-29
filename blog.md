# Predicting Board Game Type with KNN: A Beginner's Guide

![Board Game](board_game_pic.jpg "Board Game")

## Introduction

Have you ever had a hobby that you wish you could predict, without having to do any of the work? Maybe you want to be able to predict what sport an athlete plays just based on height, weight, and eye color? Or guess which actor will be cast for a role based on their previous roles? In this tutorial, we’ll walk through how to use the **K-Nearest Neighbors (KNN)** algorithm to classify data based on the attributes they possess. We will use Board Game data as an example, modeling the type of board game based of attributes like player count, time, and complexity

This guide is perfect for Data Science students who want to apply machine learning to real-world data in a reproducible and interpretable way.

---

## What You'll Learn

- How to prepare data for modeling
- How KNN works and why it's useful for classification
- How to implement KNN in Python using `scikit-learn`
- How to evaluate model performance
- How to interpret predictions

---

## Step 1: Understanding the Data

We’ll use a simplified dataset of board games with the following attributes:
'Min Players', 'Max Players', 'Play Time', 'Min Age', 'Rating Average', 'BGG Rank', 'Complexity Average'

| Name             | Min Players | Max Players | Play Time | Min Age | Rating Average | BGG Rank | Complexity Average | Type        |
|------------------|-------------|-------------|-----------|---------|----------------|----------|------------|-------------|
| Codenames        | 2           | 8           | 15        | 10      | 8.52           | 20       | 1.2        | Party       |
| Dominion         | 2           | 4           | 30        | 13      | 5.37           | 5        | 2.3        | Card Game   |
| Twilight Struggle| 2           | 2           | 180       | 13      | 2.80           | 572      | 3.8        | Strategy    |

> **Note**: Complexity is a rating from 1 (easy) to 5 (complex), sourced from [BoardGameGeek](https://boardgamegeek.com/wiki/page/Weight)

K-Nearest Neighbors is a **non-parametric**, **instance-based** learning algorithm. It classifies a new data point based on the majority label of its *k* closest neighbors in the feature space. This means that if a new board game is entered and of its *k* closest neighbors, the majority are **Strategy** games, it will classify this new game as a **Strategy** game.

### The KNN Formula


To compute the distance between two games, we use either **Euclidean distance** or **Manhattan distance**:

#### Euclidean distance
$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$
Where:
- \( x \) and \( y \) are feature vectors
- \( n \) is the number of features

#### Manhattan distance
$$
d(x, y) = \sum{i=1}^{n} |x_i - y_i|
$$
Where:
- \( x \) and \( y \) are feature vectors
- \( n \) is the number of features

We will also choose how to weight our distances, either **Uniform** or **Distance**

If we choose **Uniform**, all of the neighbors will be weighted the same, while if we choose **Distance**, the neighbors that are closest will have the biggest influence.

## Step 2: Implementing KNN in Python

Here’s how to build a simple KNN classifier using `scikit-learn`.

### Import Libraries
We start by importing the necessary Libraries

```python
# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hamming_loss, f1_score

```
These librarys will allow us to access the functions we need to run this KNN algorithm.

### Load and Prepare Data
Next we will load and prepare the data, formatting it a way that allows us to run classification

```python
data = pd.read_csv('BGG_Data_Set.csv', encoding='ISO-8859-1') #Replace the inside of the read_csv function with you file
```

After we import the data we need to decide what to do with the null values, I chose to drop all of the data that wasn't given a game type. You could also choose to replace null's with the average or mode of the data, there are many different ways to deal with null data so you choose which works best for your data set
```python
# Replace NaN values with empty strings
data['Mechanics'] = data['Mechanics'].fillna('')
data['Domains'] = data['Domains'].fillna('')

# Remove rows where 'Domains' is still empty
data = data[data['Domains'] != ""]
data = data.reset_index(drop=True)
```

Because my data has multiple types of games assigned to each game I had to split them up. First I separated each *Domain* and *Mechanics* with ',' and then used "MultiLabelBinarizer" to separate them into different columns with 1 and 0 variables. I also standardized the data to make sure that no one attribute was pulling the algorithm too much. Finally I seperated the data into two data frames, one for the data I am using to train and one with the type of game for each game.
```python
# Split strings into lists and clean them
data['Mechanics'] = data['Mechanics'].str.split(', ').apply(lambda x: [item.strip() for item in x if item.strip()])
data['Domains'] = data['Domains'].str.split(', ').apply(lambda x: [item.strip() for item in x if item.strip()])

scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(data[['Min Players', 'Max Players', 'Play Time', 'Min Age', 'Rating Average', 'BGG Rank', 'Complexity Average']])
scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=['Min Players', 'Max Players', 'Play Time', 'Min Age', 'Rating Average', 'BGG Rank', 'Complexity Average'])

# Binarize the cleaned lists
mlb = MultiLabelBinarizer()
attribute_matrix = pd.DataFrame(mlb.fit_transform(data['Mechanics']), columns=mlb.classes_)
domain_matrix = pd.DataFrame(mlb.fit_transform(data['Domains']), columns=mlb.classes_)

# Build feature matrix from the same filtered `data`
feature_matrix = pd.concat([scaled_numeric_df, attribute_matrix], axis=1)
```

### Train/Fit Algorithm
Before you train your algorithm, you first need to split your data into test and train data. This function "train_test_split" takes a data frame of the attributes and a separate one of the "target" or answers and splits them up randomly.(the test_size parameter allows you to specify how much of the full data you want to be marked test data)

```python
x_train, x_test, y_train, y_test = train_test_split(feature_matrix, domain_matrix, test_size=0.2, random_state=42)
```

This next part is optional, but if you are not sure which parameters are the best for you data, you can run a grid search to test different combinations. This function will return the combination that performs the best(it may take a while). You can change the values for n_neighbors to values you think might be best. At the end of this function the "knn_grid.fit()" function will fit your model to the data, and now you have a working model
```python
knn = MultiOutputClassifier(KNeighborsClassifier())

param_grid_knn = {
    'estimator__n_neighbors': [10,11,12],
    'estimator__weights': ['uniform', 'distance'],
    'estimator__metric': ['euclidean', 'manhattan']
}
knn_grid = GridSearchCV(knn, param_grid_knn, cv=5)
knn_grid.fit(x_train, y_train)
print("Best KNN parameters:", knn_grid.best_params_)
```


### Test Algorithm
Finally it is time to test how well our model did. There are various different metrics to test your model and I will show you a few that you can use.
Accuracy: This is just the percentage of correct guesses your alogithm made, a simple metric, but usefull
useful
Hamming loss: This is used specifically for multi-label classification, it measures the percent of labels that are incorrect(either a missed label or a wrong label). A lower score is better.

F1 Score: This is a combination of both Precision("Of all the instances my model predicted as positive, how many were actually positive?") and Recall("Of all the actual positive instances, how many did my model correctly identify?")

```python
y_pred = knn_grid.best_estimator_.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy}")
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
```

### Use Model
Now that our model has been Trained, Fit, and Tested, we can use it on further data. By using the function
```python
"model_name".predict("new_data")
```
our model will give us its prediction for this new data point. Using our example we could predict what type a new game is, just based on its mechanical attributes.


## Conclusion
Now that you have learned how to implement a Multi-Label classification, KNN model, its time to go and try it yourself. Find yourself an interesting data set and see if you can create a KNN prediction model for it. Try out using different attributes or trying to predict different targets and see how the model fairs and use the accuracy scores to see how well you did. Once you are done with this you can go on to learn other prediction algorithms and continue to expand your data science toolkit!
