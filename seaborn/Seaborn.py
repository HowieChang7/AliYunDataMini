
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 这是一款绘图包
#set_style( )
# set_style( )是用来设置主题的，Seaborn有五个预设好的主题： darkgrid , whitegrid , dark , white ,和 ticks  默认： darkgrid
# sns.set_style("darkgrid")
# plt.plot(np.arange(10))
# plt.title("darkgrid")
# plt.show()

#set( )设置主题，调色板更常用
# sns.set(style="whitegrid", palette="deep", color_codes=True)
# plt.plot(np.arange(10))
# plt.show()

# df_iris = sns.load_dataset("iris")
# fig, axes = plt.subplots(1,2)
# # hist加强版 kde 密度曲线  rug 边际毛毯
# sns.distplot(df_iris['petal_length'], ax = axes[0], color = 'b', kde = True, rug = True)
# # shade  阴影 kdeplot( )为密度曲线图
# sns.kdeplot(df_iris['petal_length'], ax = axes[1], color = 'r', shade= True)
# plt.show()

# sns.set( palette="muted", color_codes=True)
# df_iris = sns.load_dataset("iris")
# fig, axes = plt.subplots(2,2,figsize=(7, 7), sharex=True)
# # hist加强版 kde 密度曲线  rug 边际毛毯
# sns.distplot(df_iris['petal_length'], ax = axes[0,0], color = 'b', kde = False, rug = True)
# # shade  阴影 kdeplot( )为密度曲线图
# sns.distplot(df_iris['petal_length'], ax = axes[0,1], color = 'g', kde = True, rug = True, hist= False)
# # shade  阴影 kdeplot( )为密度曲线图
# sns.distplot(df_iris['petal_length'], ax = axes[1,0], color = 'r', kde = True, rug = True, hist=False)
# # shade  阴影 kdeplot( )为密度曲线图
# sns.distplot(df_iris['petal_length'], ax = axes[1,1], color = 'b', kde = True, rug = True)
# plt.show()

# df_iris = sns.load_dataset("iris")
# sns.boxplot(x = df_iris['species'],y = df_iris['petal_length'])
# plt.show()

# df_iris = sns.load_dataset("iris")
# print(df_iris.head(5))
# sns.jointplot("petal_length", "sepal_length", df_iris, kind='hex')
# plt.show()


df_iris = sns.load_dataset("iris")
print(df_iris.head(5))
plt.scatter(x = df_iris["petal_length"], y = df_iris["sepal_length"],c='r')
plt.show()
