import numpy as np

n = 3
m = 4
Anp = np.arange(0, 12).reshape(n, m)
A = Anp.tolist()
Bnp = np.arange(0, 12).reshape(m, n)
B = Bnp.tolist()
Bnp_ = np.arange(0, 12).reshape(n, m)
B_ = Bnp_.tolist()
C = [['' for _ in range(n)] for _ in range(n)]
print(Anp)
print(Bnp)
ans = np.matmul(Anp, Bnp)
print(ans)
for i in range(n):
    for j in range(m):
        for k in range(n):
            C[i][k] +=  '+' + str(A[i][j])  + '*' + str(B[j][k])
print(C)
print()

from collections import OrderedDict
d = OrderedDict()
s = set()
s.discard()