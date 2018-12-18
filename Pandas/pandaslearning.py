# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# 序列Series和数据框DataFrame

#Series的创建
# 通过array创建，默认index为0，1，2，3
# data = np.arange(10)
# print(data)
# print(type(data))
# pdata = pd.Series(data)
# print(pdata)
# print(type(pdata))

# 通过字典方式创建
# dic = {'a':1,'b':2,'c':3,'d':2,'e':1}
# pddata = pd.Series(dic)
# print(pddata)

#通过DataFrame中的某一行或某一列创建序列

# DataFrame(数据框)的创建
# 通过二维数组创建数据框
# data = np.array(np.arange(12)).reshape(3,4)
# print(data)
# print(type(data))
# df = pd.DataFrame(data)
# print(df)
# print(type(df))

# 通过字典方式创建,方向：先水平，后竖直
# dic2 = {'a':[1,2,3,4],'b':[5,6,7,8],'c':[9,10,11,12],'d':[13,14,15,16]}
# print(dic2)
# print(type(dic2))
# df2 = pd.DataFrame(dic2)
# print(df2)
# print(type(df2))
#
# dic3 = {'one':{'a':1,'b':2,'c':3,'d':4},'two':{'a':5,'b':6,'c':7,'d':8},'three':{'a':9,'b':10,'c':11,'d':12}}
# print(dic3)
# print(type(dic3))
# df3 = pd.DataFrame(dic3)
# print(df3)
# print(type(df3))
#
# #通过数据框的方式创建数据框
# df4 = df3[["one","three"]]
# print(df4)

# 数据索引 如果通过索引标签获取数据的话，末端标签所对应的值是可以返回的！在一维数组中，就无法通过索引标签获取数据，这也是序列不同于一维数组的一个方面
# s4 = pd.Series(np.array([1,1,2,3,5,8]))
# print(s4)
# print(s4.index)
# s4.index = ['a','b','c','d','e','f']
# print(s4.index)
# print(s4)
#
# print(s4['c'])
# print(s4[3])

#自动化对齐

# s5 = pd.Series(np.array([1,2,3,4,5]),index = ['a','b','c','d','e'])
# s6 = pd.Series(np.array([5,4,3,2,1,7]),index = ['a','b','c','d','e','f'])
# print(s5)
# print(s6)
# print(s5+s6)

# 利用pandas查询数据
# 里的查询数据相当于R语言里的subset功能，可以通过布尔索引有针对的选取原数据的子集、指定行、指定列等。我们先导入一个student数据集
# stu_dic = {'Age':[14,13,13,14,14,12,12,15,13,12,11,14,12,15,16,12,15,11,15],
# 'Height':[69,56.5,65.3,62.8,63.5,57.3,59.8,62.5,62.5,59,51.3,64.3,56.3,66.5,72,64.8,67,57.5,66.5],
# 'Name':['Alfred','Alice','Barbara','Carol','Henry','James','Jane','Janet','Jeffrey','John','Joyce','Judy','Louise','Marry','Philip','Robert','Ronald','Thomas','Willam'],
# 'Sex':['M','F','F','F','M','M','F','F','M','M','F','F','F','F','M','M','M','M','M'],
# 'Weight':[112.5,84,98,102.5,102.5,83,84.5,112.5,84,99.5,50.5,90,77,112,150,128,133,85,112]}
# student = pd.DataFrame(stu_dic)
# print(student)

# 前五行
# print(student.head())
# # 后五行
# print(student.tail())
# # 查询指定的行，这里的loc索引标签函数必须是中括号[]
# print(student.loc[[0,1,5,6,8]])
# # 查询指定列
# print(student[['Age','Name']])
# 按照条件查询
# print(student[(student['Sex'] == 'M') & (student['Age'] > 13)])

def stats(x):
	return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),x.quantile(.75),
                      x.mean(),x.max(),x.idxmax(),x.mad(),x.var(),x.std(),x.skew(),x.kurt()],
                     index = ['Count','Min','Whicn_Min','Q1','Median','Q3','Mean','Max',
                              'Which_Max','Mad','Var','Std','Skew','Kurt'])

# 利用pandas的DataFrames进行统计分析¶
np.random.seed(1234)
d1 = pd.Series(2*np.random.normal(size = 100)+3)
d2 = np.random.f(2,4,size = 100)
d3 = np.random.randint(1,100,size = 100)
# print(d1)
# print(stats(d1))
# print('非空元素计算: ', d1.count()) #非空元素计算
# print('最小值: ', d1.min()) #最小值
# print('最大值: ', d1.max()) #最大值
# print('最小值的位置: ', d1.idxmin()) #最小值的位置，类似于R中的which.min函数
# print('最大值的位置: ', d1.idxmax()) #最大值的位置，类似于R中的which.max函数
# print('10%分位数: ', d1.quantile(0.1)) #10%分位数
# print('20%分位数: ', d1.quantile(0.2)) #20%分位数
# print('求和: ', d1.sum()) #求和
# print('均值: ', d1.mean()) #均值
# print('中位数: ', d1.median()) #中位数
# print('众数: ', d1.mode()) #众数
# print('方差: ', d1.var()) #方差
# print('标准差: ', d1.std()) #标准差
# print('平均绝对偏差: ', d1.mad()) #平均绝对偏差
# print('偏度: ', d1.skew()) #偏度
# print('峰度: ', d1.kurt()) #峰度
# #descirbe方法只能针对序列或数据框，一维数组是没有这个方法的
# print('描述性统计指标: ', d1.describe()) #一次性输出多个描述性统计指标

#在实际的工作中，我们可能需要处理的是一系列的数值型数据框，如何将这个函数应用到数据框中的每一列呢？
# 可以使用apply函数，这个非常类似于R中的apply的应用方法。 将之前创建的d1,d2,d3数据构建数据框

# df = pd.DataFrame(np.array([d1,d2,d3]).T,columns=['x1','x2','x3'])
# print(df)
# # print(df.apply(stats))
#
# # 对于离散数据，用describe方法就可以实现这样的统计
# # 连续变量的相关系数（corr）和协方差矩阵（cov）
# print(df.corr())
# print(df.cov())
# # 关于相关系数的计算可以调用pearson方法或kendell方法或spearman方法，默认使用pearson方法。
# print(df.corr('spearman'))
# # 如果只想关注某一个变量与其余变量的相关系数的话，可以使用corrwith,如下方只关心x1与其余变量的相关系数
# print(df.corrwith(df['x1']))

#pandas 实现增删改查

# stu_dic = {'Age':[14,13,13,14,14,12,12,15,13,12,11,14,12,15,16,12,15,11,15],
# 'Height':[69,56.5,65.3,62.8,63.5,57.3,59.8,62.5,62.5,59,51.3,64.3,56.3,66.5,72,64.8,67,57.5,66.5],
# 'Name':['Alfred','Alice','Barbara','Carol','Henry','James','Jane','Janet','Jeffrey','John','Joyce','Judy','Louise','Marry','Philip','Robert','Ronald','Thomas','Willam'],
# 'Sex':['M','F','F','F','M','M','F','F','M','M','F','F','F','F','M','M','M','M','M'],
# 'Weight':[112.5,84,98,102.5,102.5,83,84.5,112.5,84,99.5,50.5,90,77,112,150,128,133,85,112]}
# student = pd.DataFrame(stu_dic)
# # print(student)
# dic = {'Name':['LiuShunxiang','Zhangshan'],'Sex':['M','F'],'Age':[27,23],'Height':[165.7,167.2],'Weight':[61,63]}
# student2 = pd.DataFrame(dic)
# # print(student2)
#
# # 将student2 中的数据添加到 student中
# student3 = pd.concat([student,student2],sort=False)
# print(pd.DataFrame(student2,columns=['Age','Height','Name','Sex','Weight','Score']))
# print(student2)

# 删除数据框  del student2
# 删除指定的行 student.drop([0,1,3,6])
# 聚合：pandas模块中可以通过groupby()函数实现数据的聚合操作

# 排序
# Data = pd.Series(np.array(np.random.randint(1,20,10)))
# print(Data)
# print(Data.sort_index())
# print(Data.sort_values(ascending=False))
# 多表连接
# stu_score1 = pd.merge(student, score, on='Name')
# stu_score2 = pd.merge(student, score, on='Name', how='left')


# 利用pandas进行缺失值的处理¶
