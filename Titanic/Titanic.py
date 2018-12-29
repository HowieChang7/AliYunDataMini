# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.core.umath_tests import inner1d
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sns.set_style('dark')
# print(train.info())
# print(test.info())
# plt.show(train["Survived"].value_counts().plot.pie(autopct = '%1.2f%%'))
# 填充缺失值，Embarked字段有两个为null,使用众数填充
train.Embarked[train.Embarked.isnull()] = train.Embarked.dropna().mode().values
# 对于标称型数据，可以设置一个默认值来代替空值
train["Cabin"] = train.Cabin.fillna("U0")
# print(train.info())

# 对于age缺失值，因为age是一个比较重要的特征，我们需要保证一定的预测准确性
#一般情况下，会使用数据完整的条目作为模型的训练集，以此来预测缺失值。
#我们在这里使用随机森林法进行预测
# 将数值数据提取出来
from sklearn.ensemble import  RandomForestRegressor
age_df = train[["Age","Survived","Pclass","SibSp","Parch","Fare"]]
age_notnull = age_df.loc[(age_df["Age"].notnull())]
age_isnull = age_df.loc[(age_df["Age"].isnull())]
X = age_notnull.values[:,1:]
Y = age_notnull.values[:,0]

# use RandomForestRegressor traindata
# n_jobs：这个参数告诉引擎有多少处理器是它可以使用。 “-1”意味着没有限制，而“1”值意味着它只能使用一个处理器。
# n_estimators：在利用最大投票数或平均值来预测之前，你想要建立子树的数量。 较多的子树可以让模型有更好的性能，但同时让你的代码变慢。
RTF = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
RTF.fit(X,Y)
predictAges = RTF.predict(age_isnull.values[:,1:])
train.loc[train['Age'].isnull(), ['Age']] = predictAges

# 判断性别与是否生存的关系，结论是女性存活比例远远高于男性
# print(train.groupby(['Sex','Survived'])['Survived'].count())
# plt.show(train[["Survived","Sex"]].groupby(["Sex"]).mean().plot.bar())

# 判断船舱等级与是否生存的关系，结论是船舱等级越高存活率越高
# print(train.groupby(['Pclass','Survived'])['Survived'].count())
# plt.show(train[["Survived","Pclass"]].groupby(["Pclass"]).mean().plot.bar())

# 判断船舱等级，性别与存活率的关系 # 女士优先是主流，富人优先也是主流
# print(train.groupby(['Sex','Pclass','Survived'])['Survived'].count())
# plt.show(train[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar())

# 年龄分布 bins 指的是 宽度
# 年龄，舱等级，性别与存活之间的关系
# fig, ax = plt.subplots(1, 3, figsize = (18, 8))
# sns.violinplot("Pclass", "Age", hue="Survived", data=train, split=True, ax=ax[0])
# ax[0].set_title('Pclass and Age vs Survived')
# ax[0].set_yticks(range(0, 110, 10))
#
# sns.violinplot("Sex", "Age", hue="Survived", data=train, split=True, ax=ax[1])
# ax[1].set_title('Sex and Age vs Survived')
# ax[1].set_yticks(range(0, 110, 10))
#
# train["Age"].hist(bins=50, ax=ax[2])
# plt.show()

# 年龄分布
# plt.figure(figsize=(12,5))
# plt.subplot(121)
# train['Age'].hist(bins=70)
# plt.xlabel('Age')
# plt.ylabel('Num')
#
# plt.subplot(122)
# train.boxplot(column='Age', showfliers=False)
# plt.show()

#不同年龄下的生存和非生存的分布情况：
# facet = sns.FacetGrid(train, hue="Survived",aspect=4)
# facet.map(sns.kdeplot,'Age',shade= True)
# facet.set(xlim=(0, train['Age'].max()))
# facet.add_legend()
# plt.show(facet)

# 不同年龄下的平均生存率
# fig, axis1 = plt.subplots(1,1,figsize=(18,4))
# train["Age_int"] = train["Age"].astype(int)
# ave_age = train[["Age_int","Survived"]].groupby(["Age_int"],as_index=False).mean()
# # sns.barplot(x='Age_int', y='Survived', data=ave_age)
# plt.show(sns.barplot(x='Age_int', y='Survived', data=ave_age))

#
print(train["Age"].describe())
bins = [0,12,18,65,100]
train["Age_group"] = pd.cut(train["Age"],bins)
by_age = train.groupby(["Age_group"])["Survived"].mean()
print(by_age)
by_age.plot(kind = "bar")
plt.show()