# -*- coding: utf-8 -*- 
# @since : 2020/5/18 15:31 
# @author : wongleon
x = [0, 1, 2, 3, -4, 0, 6]
i = len(x) - 1
j = i
while i > 0:
    if x[i] != 0:
        x[j] = x[i]  # 将非零元素往后挪
        j -= 1
    i -= 1
while j > 0:
    x[j] = 0
    j -= 1

print x
