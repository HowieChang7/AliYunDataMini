# -*- coding: utf-8 -*-
import numpy as np

#原始数据
X = [1,2,3,4,5,6]
Y=[ 2.6 ,3.4 ,4.7 ,5.5 ,6.47 ,7.8]

#用一次多项式拟合，相当于线性拟合
z1 = np.polyfit(X, Y, 1)
p1 = np.poly1d(z1)
print (z1)  #[ 1.          1.49333333]
print (p1)  # 1 x + 1.493

#作图显示
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1,7)
y = z1[0] * x + z1[1]
#创建一个窗口
plt.figure()
#绘制原始数据散点图
plt.scatter(X, Y)
#绘制拟合后的图
plt.plot(x, y)
plt.show()