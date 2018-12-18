# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from matplotlib import style

X = [5,2,7]
Y = [2,16,4]
# 折线图
# plt.plot(X,Y)
# plt.title('chart')
# plt.ylabel("Y")
# plt.xlabel("X")
# plt.show()

#
# style.use('ggplot')
# x = [5,8,10]
# y = [12,16,6]
# x2 = [6,9,11]
# y2 = [6,15,7]
# plt.plot(x,y,'g',label = 'line one',linewidth=5)
# plt.plot(x2,y2,'r',label='line two',linewidth=5)
# plt.legend()   #标签
# # plt.grid(True,color='k') # 分割线
# plt.show()

# 条形图使用条形来比较不同类别之间的数据。当您想要测量一段时间内的变化时
plt.bar([0.25,1.25,2.25,3.25,4.25],[50,40,70,80,20],label='BMW',color='b',width=0.5)
plt.bar([0.75,1.75,2.75,3.75,4.75],[70,80,80,90,30],label='AUTI',color='r',width=0.5)
plt.title("business")
plt.legend()
plt.show()