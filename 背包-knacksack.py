# -*- coding: utf-8 -*- 
# @since : 2020-03-07 17:54 
# @author : wongleon

N = 6  # 物品的数量，
W = 10  # 书包能承受的重量，
w = [2, 2, 3, 1, 5, 2]  # 每个物品的重量，
v = [2, 3, 1, 5, 4, 3]  # 每个物品的价值

c = [[0 for j in range(W + 1)] for i in range(N + 1)]
for i in range(1, N + 1):
    for j in range(1, W + 1):
        if j < w[i - 1]:
            c[i][j] = c[i - 1][j]
        else:
            c[i][j] = max(c[i - 1][j], c[i - 1][j - w[i - 1]] + v[i - 1])
print c[N][W]

Item = [False for i in range(N + 1)]

j = W
for i in range(N, 0, -1):
    if (c[i][j] > c[i - 1][j]):
        Item[i - 1] = True
        j -= w[i - 1]

print '最优选择为：第' + ', '.join([str(i + 1) for i in range(N) if Item[i]]) + '个物品'
