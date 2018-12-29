# -*- coding: utf-8 -*-
import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
# fruits = pd.DataFrame([[30,21]],columns = ['Apples','Bananas'])
# print(fruits)

# quality = ['4 cups','1 cup','2 large','1 can']
# index = ['Flour','Milk','Eggs','Spam']
# ingredients = pd.Series(quality,index=index,name='Dinner')
# print(ingredients)

# 第一列为index列
# reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)

# 数据
# import sqlite3
# conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
# music_reviews = pd.read_sql_query("SELECT * FROM artists", conn)

#In general, when we select a single column from a DataFrame, we'll get a Series.

# p = pd.Series().sort_values()


ro