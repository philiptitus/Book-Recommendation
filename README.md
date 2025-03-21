# Book Recommendation System using Collaborative Filtering

## Overview
This repository implements a collaborative filtering recommendation system for books using TensorFlow and the Goodreads dataset.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Technical Details](#technical-details)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Features
- Matrix factorization-based collaborative filtering
- Handles sparse rating matrices
- Personalized book recommendations
- Interactive visualization tools
- Custom user ratings support
- Synthetic data generation for cold-start cases

## Installation
```bash
# Clone repository
git clone https://github.com/philiptitus/book-recommender.git
cd book-recommender

# Install dependencies
pip install numpy tensorflow pandas matplotlib seaborn scikit-learn
```

## Technical Details
### Data Structure
```python
# Rating Matrix (Y): User-book ratings (1-5)
# Binary Matrix (R): Rating indicators (0/1)
# Feature Matrices: W (users) and X (books)
num_features = 100
num_books = Y.shape[0]
num_users = Y.shape[1]
```

### Model Parameters
```python
W = tf.Variable(tf.random.normal((num_users, num_features)))
X = tf.Variable(tf.random.normal((num_books, num_features)))
b = tf.Variable(tf.random.normal((1, num_users)))
```

## Usage
### Data Preparation
```python
# Load dataset
books_df = pd.read_csv('books.csv')
Y, R = create_ratings_matrices(books_df)

# Initialize ratings
my_ratings = np.zeros(num_books)
```


## Model Architecture
### Collaborative Filtering Implementation
```python
def predict_ratings(X, W, b):
    """
    X: Book features matrix
    W: User features matrix
    b: Bias terms
    """
    return tf.matmul(X, tf.transpose(W)) + b
```

### Cost Function
```python
def compute_cost(X, W, b, Y, R, lambda_reg):
    """
    Regularized cost function
    """
    predictions = predict_ratings(X, W, b)
    regularization = (lambda_reg/2) * (tf.reduce_sum(tf.square(X)) + 
                                     tf.reduce_sum(tf.square(W)))
    return tf.reduce_mean(
        tf.square(tf.multiply((predictions - Y), R))
    ) + regularization
```

## Visualization
### Learning Curves
```python
def plot_learning_curve(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title('Model Learning Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()
```

### Rating Distribution
```python
def plot_ratings(Y):
    plt.hist(Y[Y != 0].flatten(), bins=20)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()
```

### Performance Metrics
```python
def evaluate_model(Y_true, Y_pred, R):
    mask = R == 1
    rmse = np.sqrt(np.mean(((Y_pred - Y_true)[mask]) ** 2))
    mae = np.mean(np.abs((Y_pred - Y_true)[mask]))
    return {'RMSE': rmse, 'MAE': mae}
```



## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License
MIT License

```
Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files.
