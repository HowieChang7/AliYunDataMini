# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
math_students = pd.read_csv("student-mat.csv",sep=";");
por_students = pd.read_csv("student-por.csv",sep=";")
# print(math_students.head(10))
# print(por_students.head(10))
# print(math_students.shape)
# print(por_students.shape)

# 将两个数据集合并
all_students = pd.concat([math_students,por_students])
# print(all_students.shape)

# 检查每一个特征中空值的情况
# print(all_students.isnull().sum())

# 查看每一个特征中特征的类型
# print(all_students.dtypes)

# 查看数据统计信息
print(all_students.describe())  # 数值特征
# print(all_students.describe(include = 'all'))  # 全部特征