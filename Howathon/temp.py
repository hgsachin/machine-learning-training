import pandas as pd
import numpy as nm
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

Product_data = pd.read_csv('Prod_Data.csv', encoding='latin-1')

Mean = Product_data.groupby(['UserId'], as_index= False, sort = False).mean().rename(columns={"UserId":"UserId", "ProductId":"ProductId", "Ratings":"Ratings_Mean"})

Product_data= pd.merge(Product_data, Mean, on='UserId', how="left", sort="False")

Product_data['ratings_adjusted'] = Product_data["Ratings"]- Product_data["Ratings_Mean"]


result = pd.DataFrame({"UserId":Product_data['UserId'],
                       "ProductId":Product_data['ProductId_x'],
                       "rating":Product_data['ratings_adjusted']})

result1= result.pivot_table(index ='UserId', columns ='ProductId', values='rating').fillna(0)


result1.to_csv('Prod_Data_updated.csv')

all_users = result1.values
A_sparse = sparse.csr_matrix(all_users)
similarities = cosine_similarity(A_sparse)
pd.DataFrame(similarities).to_csv('User_Data.csv', index = 'UserId')