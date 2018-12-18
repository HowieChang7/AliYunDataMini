# -*- coding: utf-8 -*-
import numpy as np

# 用于数据整理和清理、子集构造和过滤、转换等快速的矢量化数组运算。
# 常用的数组算法，如排序、唯一化、集合运算等。
# 高效的描述统计和数据聚合/摘要运算。
# 用于异构数据集的合并/连接运算的数据对齐和关系型数据运算。
# 将条件逻辑表述为数组表达式（而不是带有if-elif-else分支的循环）。
# 数据的分组运算（聚合、转换、函数应用等）。。

#NumPy之于数值计算特别重要的原因之一，是因为它可以高效处理大数组的数据。

# my_arr = np.arange(1000000)
# my_list = list(range(1000000))

# NumPy的ndarray：一种多维数组对象
# data = np.random.randn(2,3)
# print(data)
# print("data * 10 \n", data*10)
# print("data + data \n", data+data)

# ndarray所有元素必须是相同类型的。每个数组都有一个shape（一个表示各维度大小的元组）和一个dtype（一个用于说明数组数据类型的对象）
# print("data shape:" , data.shape)
# print("data dtype:" , data.dtype)

# data = [6, 7.5, 8, 0, 1]
# arr = np.array(data)
# print(data)

# 嵌套序列（比如由一组等长列表组成的列表）将会被转换为一个多维数组：
# 上述data是列表的列表，arr是数组
# data = [[1,2,3],[4,5,6]]
# arr = np.array(data)
# print(data)
# print(arr)
# print(arr.ndim)
# print(arr.shape)
# print(arr.dtype)
# 但zeros并不会返回全零值，它返回的都是一些未初始化的垃圾值。
# print(np.zeros(10))
# print(np.zeros((3, 6)))
# print(np.empty((2, 3, 2)))

#改变数据类型
# data = [1,2,3,4,5]
# arr = np.array(data)
# print(arr.dtype)
# float_arr = arr.astype("float64")
# print(float_arr.dtype)
# print(arr)
# print(float_arr)
#
# data1 = [1.3,3.4,9.7]
# arr1 = np.array(data1)
# print(arr1)
# int_arr = arr1.astype("int32")
# print(int_arr)

# int_array = np.arange(10)
# calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
# print(int_array.astype(calibers.dtype))

#NumPy数组的运算
# arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# print(arr)
# print(arr + arr)
# # arr乘法是对应位置数据的乘法
# print(arr * arr)
# print(1/arr)
# print(arr**0.5)

#数组的广播
# 如果两个数组的维数不相同，则元素到元素的操作是不可能的。 然而，在 NumPy 中仍然可以对形状不相似的数组进行操作，因为它拥有广播功能。 较小的数组会广播到较大数组的大小，以便使它们的形状可兼容。
#
# 如果满足以下规则，可以进行广播：
#
# ndim较小的数组会在前面追加一个长度为 1 的维度。
# 输出数组的每个维度的大小是输入数组该维度大小的最大值。
# 如果输入在每个维度中的大小与输出大小匹配，或其值正好为 1，则可以在计算中使用该输入。
# 如果输入的某个维度大小为 1，则该维度中的第一个数据元素将用于该维度的所有计算。
# 如果上述规则产生有效结果，并且满足以下条件之一，那么数组被称为可广播的。
#
# 数组拥有相同形状。
# 数组拥有相同的维数，每个维度拥有相同长度，或者长度为 1。
# 数组拥有极少的维度，可以在其前面追加长度为 1 的维度，使上述条件成立。

# a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]])
# b = np.array([1.0,2.0,3.0])
#
# print(a)
# print(b)
# print(a+b)

#基本的索引和切片
# arr = np.arange(10)
# print(arr)
#
# print(arr[5])
# print(arr[3:])
# print(arr[:3])
# print(arr[2:5])
# arr[6:] = 888
# print(arr)
# #数组切片是原始数组的视图。这意味着数据不会被复制，视图上的任何修改都会直接反映到源数组上。
# arr_slice = arr[5:7]
# print(arr_slice)
# arr_slice[1] = 78
# print(arr_slice)
# print(arr)

#如果你想要得到的是ndarray切片的一份副本而非视图，就需要明确地进行复制操作，例如arr[5:8].copy()。
# arr1 = np.arange(10)
# print(arr1)
# arr1_slice = arr1[3:8].copy()
# print(arr1_slice)
# arr1_slice[1] = 777
# print(arr1)
# print(arr1_slice)

#切片
# arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(arr2d)
# print(arr2d[:2])
# print(arr2d[1:])
# print(arr2d[:2,1:])

#花式索引
#花式索引（Fancy indexing）是一个NumPy术语，它指的是利用整数数组进行索引。花式索引是数据复制，不是数据视图
# arr = np.empty((8, 4))
# for i in range(8):
#   arr[i] = i
# print (arr)
# print(arr[[0,5,1,3]])

#数组转置和轴对换
#转置是重塑的一种特殊形式，它返回的是源数据的视图（不会进行任何复制操作）。数组不仅有transpose方法，还有一个特殊的T属性：
# arr = np.arange(6).reshape((2, 3))
# print(arr)
# print(arr.T)
# # print(arr.transpose())
# print(np.dot(arr,arr.T))

#对于高维数组，transpose需要得到一个由轴编号组成的元组才能对这些轴进行转置
# arr = np.arange(16).reshape((2, 2, 4))
# print(arr)
# print('==========分割线==============')
# print(arr.transpose((1, 0, 2)))


#通用函数：快速的元素级数组函数¶
# arr = np.arange(4)

# print(arr)
# print(np.sqrt(arr))
# print(np.exp(arr))

# x = np.arange(4)
# y = np.arange(4)
#
# print(x)
# print(y)
# print(np.maximum(x, y*2))

# arr = np.random.randn(4) * 5
# print(arr)
# # modf将浮点数的整数部分和小数部分分开
# remainder, whole_part = np.modf(arr)
# print(remainder)
# print(whole_part)

# 利用数组进行数据处理
# points = np.arange(1, 3, 1) # 1000 equally spaced points
# print(points)
# xs, ys = np.meshgrid(points, points)
# print(ys)
# print(xs ** 2)

#zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
# xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
# yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
# cond = np.array([True, False, True, True, False])
#
# result = np.where(cond, xarr, yarr)
# print(result)


#数学和统计方法¶
# arr = np.arange(4).reshape((2, 2))
# print(arr)
# print(arr.mean())
# print(np.mean(arr))
# print(arr.sum())
#
# # axis表示数据处理的方向
# print(arr.mean(axis=0))
# print(arr.mean(axis=1))

#cumsum 前几个数据的和 cumprod 累计求积
# arr = np.array([1, 2, 3, 4, 5, 6, 7])
# print(arr)
# print(arr.cumsum())
# print(arr.cumprod())

#在多维数组中，累加函数（如cumsum）返回的是同样大小的数组，但是会根据每个低维的切片沿着标记轴计算部分聚类
# axis = 0 从上到下 axis = 1 从左到右
# arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# print(arr)
# print(arr.cumsum(axis=0))
# print(arr.cumprod(axis=1))

#排序
# numpy.sort(a, axis, kind, order)
# a 数组
# axis 方向
# kind 排序类型
# order 是否以某个字段排序
# a = np.array([[3,7],[9,1]])
# print ('我们的数组是：')
# print (a)
# print ('\n')
# print ('调用 sort() 函数：')
# print (np.sort(a))
# print ('\n')
# print ('沿轴 0 排序：')
# print (np.sort(a, axis =  0))
# print ('\n')
# # 在 sort 函数中排序字段
# dt = np.dtype([('name',  'S10'),('age',  int)])
# a = np.array([("raju",21),("anil",25),("ravi",  17),  ("amar",27)], dtype = dt)
# print ('我们的数组是：')
# print (a)
# print ('\n')
# print ('按 name 排序：')
# print (np.sort(a, order =  'name'))

# arr = np.arange(10)
# # 数组是以未压缩的原始二进制格式保存在扩展名为.npy的文件中的：
# print(arr)
# np.save('some_array', arr)
# b = np.load('some_array.npy')
# print(b)
#
# #通过np.savez可以将多个数组保存到一个未压缩文件中，将数组以关键字参数的形式传入即可
# np.savez('array_archive.npz', a=arr, b=arr)
# arch = np.load('array_archive.npz')
# print(arch['b'])
#
# #如果要将数据压缩，可以使用numpy.savez_compressed：
# np.savez_compressed('arrays_compressed.npz', a=arr, b=arr)

# X = np.random.randn(2, 2)
# print(X)
# mat = np.matrix (X)
# print(mat)
#
# print(np.diag(mat))
# print(np.trace(mat))
# print(np.linalg.qr(mat))
# print(np.linalg.svd(mat))

import random
import matplotlib.pyplot as plt

position = 0
walk = [position]
steps = 1000
for i in range(steps):
     step = 1 if random.randint(0, 1) else -1
     position += step
     walk.append(position)

