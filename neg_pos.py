# -*- coding: utf-8 -*- 
# @since : 2020/5/18 16:33 
# @author : wongleon

x = [1, 2, 5, -6, -3, -7, -9, -33, 66, 99]
n = len(x) - 1
i, j = 0, 0
pos = j
while i < n:
    if x[i] < 0:
        tmp = x[i]
        x[i] = x[j]
        x[j] = tmp
        pos = j
        while pos < i: #保留j与i之间的正数的顺序
            pos += 1
            tmp = x[i]
            x[i] = x[pos]
            x[pos] = tmp
        j += 1
    i += 1

print x
