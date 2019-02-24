import numpy as np
import pandas as pd
import math

Prod_Ratings = pd.read_csv('user_product_ratings.csv', encoding="ISO-8859-1")
products = pd.read_csv('products.csv', encoding="ISO-8859-1")

#Calculate Mean rating
mean_rating = Prod_Ratings.groupby(['productId'], as_index=False, sort=False).mean().rename(columns={'rating':'rating_mean'})[['productId','rating_mean']]
#Merge the mean data to prod_rating data
Prod_Ratings = pd.merge(Prod_Ratings, mean_rating, on='productId', how='left', sort=False)
#Add new column rating_adjusted which is (user rating - mean rating)
Prod_Ratings['rating_adjusted']=Prod_Ratings['rating']-Prod_Ratings['rating_mean']

#Calculate ratings for 20 products which user has not rated which are similar to other products which user has rated
prod_data_all_append = pd.DataFrame()
distinct_users = np.unique(Prod_Ratings['userId'])
user_count=1
#For each user
for user in distinct_users:
	print(user_count, 'out of ', len(distinct_users), ' Users')
	#Get product details which are not rated by user 320
	non_user_data = Prod_Ratings[Prod_Ratings['userId']!=user]
	#Get Unique products which are not rated by user and iterate over them
	distinct_non_user_products = np.unique(non_user_data['productId'])
	i=1
	#For each item which are not rated by the user
	for product in distinct_non_user_products:
		if i%10 == 0:
			print(i, 'out of ', len(distinct_non_user_products), ' porducts')
		prod_data_all = pd.DataFrame()
		prod_data = Prod_Ratings[Prod_Ratings['productId']==product]
		prod_data = prod_data[['userId','productId','rating_adjusted']].drop_duplicates()
		prod_data=prod_data.rename(columns={'rating_adjusted':'rating_adjusted1'})
		prod_data=prod_data.rename(columns={'productId':'product1'})
		#Non user product rating  -  √Σ²
		prod1_val=np.sqrt(np.sum(np.square(prod_data['rating_adjusted1']), axis=0))
		
		user_data= Prod_Ratings[Prod_Ratings['userId']==user]
		distinct_user_products=np.unique(user_data['productId'])
		
		#For each product which are rated by the user
		for user_product in distinct_user_products:
			prod_data1 = Prod_Ratings[Prod_Ratings['productId']==user_product]
			prod_data1 = prod_data1[['userId','productId','rating_adjusted']].drop_duplicates()
			prod_data1=prod_data1.rename(columns={'rating_adjusted':'rating_adjusted2'})
			prod_data1=prod_data1.rename(columns={'productId':'product2'})
			#User product rating
			prod2_val=np.sqrt(np.sum(np.square(prod_data1['rating_adjusted2']), axis=0))
			
			#Merge two products and their ratings by keeping the User as reference
			prod_data_merge = pd.merge(prod_data,prod_data1[['userId','product2','rating_adjusted2']],on = 'userId', how = 'inner', sort = False)
			
			#vector_product = product1 rating * product2 rating
			prod_data_merge['vector_product']=(prod_data_merge['rating_adjusted1']*prod_data_merge['rating_adjusted2'])
			
			prod_data_merge= prod_data_merge.groupby(['product1','product2'], as_index = False, sort = False).sum()
			
			prod_data_merge['dot']=prod_data_merge['vector_product']/(prod1_val*prod2_val)
			
			prod_data_all = prod_data_all.append(prod_data_merge, ignore_index=True)
			
		prod_data_all=  prod_data_all[prod_data_all['dot']<1]
		prod_data_all = prod_data_all.sort_values(['dot'], ascending=False)
		prod_data_all = prod_data_all.head(20)
		
		prod_data_all_append = prod_data_all_append.append(prod_data_all, ignore_index=True)
		i=i+1
	user_count=user_count+1

result_data = prod_data_all_append[['prodId1', 'prodId2']]
pd.DataFrame(result_data).to_csv('similar_products.csv', index = 'UserId')