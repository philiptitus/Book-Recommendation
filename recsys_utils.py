import numpy as np
import pandas as pd
from numpy import loadtxt

def normalizeRatings(Y, R):
    """
    Normalize Y by subtracting the mean of each movie's ratings
    Args:
      Y (ndarray (num_movies,num_users)): The utility matrix
      R (ndarray (num_movies,num_users)): Indicator matrix for Y
    Returns:
      Ynorm (ndarray (num_movies,num_users)): Normalized Y utility matrix
      Ymean (ndarray (num_movies,1)): Mean rating for each movie
    """
    Ymean = (np.sum(Y*R, axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return(Ynorm, Ymean)


def load_precalc_params_small():

    file = open('./data/small_movies_X.csv', 'rb')
    X = loadtxt(file, delimiter = ",")

    file = open('./data/small_movies_W.csv', 'rb')
    W = loadtxt(file,delimiter = ",")

    file = open('./data/small_movies_b.csv', 'rb')
    b = loadtxt(file,delimiter = ",")
    b = b.reshape(1,-1)
    num_movies, num_features = X.shape
    num_users,_ = W.shape
    return(X, W, b, num_movies, num_features, num_users)
    
def load_ratings_small():
    file = open('./data/small_movies_Y.csv', 'rb')
    Y = loadtxt(file,delimiter = ",")

    file = open('./data/small_movies_R.csv', 'rb')
    R = loadtxt(file,delimiter = ",")
    return(Y,R)

def create_user_ratings_matrices(books_df, n_synthetic_users=100):
    n_books = len(books_df)
    
    # Create matrices
    Y = np.zeros((n_books, n_synthetic_users))
    R = np.zeros((n_books, n_synthetic_users))
    
    for book_idx in range(n_books):
        avg_rating = books_df.loc[book_idx, 'average_rating']
        n_ratings = books_df.loc[book_idx, 'ratings_count']
        
        if pd.notna(avg_rating) and n_ratings > 0:
            # Generate synthetic ratings around the average rating
            # for a random subset of users
            n_users_who_rated = min(int(n_ratings/100), n_synthetic_users)  # Scale down the number
            rating_users = np.random.choice(n_synthetic_users, n_users_who_rated, replace=False)
            
            # Generate ratings with some noise around the average
            synthetic_ratings = np.random.normal(avg_rating, 0.5, n_users_who_rated)
            # Clip ratings to be between 1 and 5
            synthetic_ratings = np.clip(synthetic_ratings, 1, 5)
            
            # Assign the ratings
            Y[book_idx, rating_users] = synthetic_ratings
            R[book_idx, rating_users] = 1
    
    return Y, R




def load_Book_List_pd():
    """ returns df with and index of movies in the order they are in in the Y matrix """
    df = pd.read_csv('books.csv', header=0, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return(mlist, df)




