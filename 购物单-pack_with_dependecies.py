# -*- coding: utf-8 -*- 
# @since : 2020-03-05 23:14 
# @author : wongleon
N, m = map(int, raw_input().split())
# N, m = 1000, 5
v = []  # [800, 400, 300, 400, 500]  # []  # price
p = []  # [2, 5, 5, 3, 2]  # []  # importance
q = []  # [0, 1, 1, 0, 0]  # []  # relation
for i in range(m):
    x, y, z = map(int, raw_input().split())
    v.append(x)
    p.append(y)
    q.append(z)

c = [[0 for j in range(N + 1)] for i in range(m + 1)]
flag = [[False for j in range(N + 1)] for i in range(m + 1)]

for i in range(1, m + 1):
    for j in range(1, N + 1):
        if q[i - 1] == 0:
            c[i][j] = c[i - 1][j]
            if j >= v[i - 1] and c[i - 1][j] <= c[i - 1][j - v[i - 1]] + v[i - 1] * p[i - 1]:
                c[i][j] = c[i - 1][j - v[i - 1]] + v[i - 1] * p[i - 1]
                flag[i][j] = True
        else:
            c[i][j] = c[i - 1][j]
            if j >= v[i - 1] and c[i - 1][j] <= c[i - 1][j - v[i - 1]] + v[i - 1] * p[i - 1] and flag[q[i - 1]][j - v[i - 1]]:
                c[i][j] = c[i - 1][j - v[i - 1]] + v[i - 1] * p[i - 1]

print c[m][N]
# for i in range(m + 1):
#     try:
#         print  "{},{}".format(i, flag[i].index(True))
#     except:
#         pass
