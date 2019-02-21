import numpy as np
import pandas as pd
import math

ratings = pd.read_csv('ratings.csv', encoding="ISO-8859-1")
movies = pd.read_csv('movies.csv', encoding="ISO-8859-1")
tags = pd.read_csv('tags.csv', encoding="ISO-8859-1")

mean = ratings.groupby(['movieId'], as_index=False, sort=False).mean().rename(columns={'rating':'rating_mean'})[['movieId','rating_mean']]
ratings = pd.merge(ratings, mean, on='movieId', how='left', sort=False)
ratings['rating_adjusted']=ratings['rating']-ratings['rating_mean']
