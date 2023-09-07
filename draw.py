#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5)
softplus = np.log(1 + np.exp(x))
sigmoid = 1 / (1 + np.exp(-x))
identity = x

logloss = np.log(1 + np.exp(x))
sample = np.power(x, 0.75)

y=np.arange(10, 100, 10)
# y=6.78 - 14.43 * np.log(odds)
odds = np.exp((6.78 - y)/14.43)
print(y)
print(np.round(odds * 100, 2))
plt.figure()
# ax = plt.gca()#获取当前坐标的位置
# ax.spines['right'].set_color('None')
# ax.spines['top'].set_color('None')
# #指定坐标的位置
# ax.xaxis.set_ticks_position('bottom') # 设置bottom为x轴
# ax.yaxis.set_ticks_position('left') # 设置left为x轴
# ax.spines["bottom"].set_position("center")
# ax.spines["left"].set_position("center")
# plt.plot(x, sigmoid)
# plt.plot(x, identity)
# plt.plot(x, softplus)
# plt.legend(['sigmoid', 'identity', 'softplus'])
plt.plot(y, odds)

plt.grid()
# #去掉坐标图的上和右 spine翻译成脊梁
# ax.axvline(5, linewidth=2, color='k')
# ax.axvline(-5, linewidth=2, color='k')
# ax.axhline(5, linewidth=2, color='k')
# ax.axhline(-5, linewidth=2, color='k')
# plt.xlim([0, 5])
# plt.ylim([-5, 5])
plt.show()
