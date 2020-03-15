# -*- coding: utf-8 -*- 
# @since : 2020-03-03 17:23 
# @author : wongleon
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(-x) - np.exp(x)) / (np.exp(-x) + np.exp(x))


def sigmoid_curve(x):
    return sigmoid(x * 4.5 + 6.5)


def tanh_curve(x):
    return tanh(2 * x - 1)


x = fsolve(lambda x: sigmoid_curve(x) - tanh_curve(x), 0)
print  "{}, {}, {}".format(x, sigmoid_curve(x), tanh_curve(x))

x1 = np.linspace(-10, -2, 50)
s1 = 0.5 * np.ones(50)
x2 = np.linspace(-2, 0, 50)
s2 = sigmoid(x2 * 4.5 + 6.5)
x3 = np.linspace(-2, 10, 50)
s3 = tanh(2 * x3 - 1)

t1 = -2
s21 = sigmoid(t1 * 4.5 + 6.5)
s31 = tanh(2 * t1 - 1)

t2 = 0
s22 = sigmoid(t2 * 4.5 + 6.5)
s32 = tanh(2 * t2 - 1)

t3 = 0.5
s33 = tanh(2 * t3 - 1)

t4 = 1
s34 = tanh(2 * t4 - 1)

# plt.plot(t1, s21, t1, s31, t2, s22, t2, s32)
# plt.legend(
#     ['(%s, %s)'.format(t1, s21), '(%s, %s)'.format(t1, s31), '(%s, %s)'.format(t2, s22), '(%s, %s)'.format(t2, s32)])
# print "({}, {})".format(t1, s21)
plt.plot(x1, s1, x2, s2, x3, s3)
plt.plot(t1, s21, '*', t1, s31, '.', t2, s22, '^', t2, s32, '+', t3, s33, 'o', t4, s34, 's', x[0], sigmoid_curve(x)[0],
         'bx')
plt.legend(['y=0.5', 'y=sigmoid(4.5*x + 6.5)', 'y=tanh(2*x - 1)'], fontsize=8)

points = [(t1, round(s21, 5)),
          (t2, round(s32, 5)),
          (t3, round(s33, 5)),
          (t4, round(s34, 5))]
for (x_, y_) in points:
    plt.text(x_, y_, (x_, y_), ha='center', va='top')

plt.text(t2, round(s22, 3), (t2, round(s22, 3)), ha='left', va='bottom')
plt.text(t1, round(s31, 3), (t1, round(s31, 3)), ha='center', va='bottom')
a = round(x[0], 2)
b = round(sigmoid_curve(x)[0], 5)
plt.text(a, b, (a, b), ha='center', va='top')

plt.grid()
plt.show()

# while True:
#     try:
#         x = input()
#         print sigmoid(x), tanh(2 * x - 1)
#     except:
#         break
