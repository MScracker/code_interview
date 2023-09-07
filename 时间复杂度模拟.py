import math
import sys
import timeit
import time


def time_cost(func):
    def helper(*args):
        start = time.time()
        func(*args)
        end = time.time()
        print("time cost:", end - start)

    return helper


@time_cost
def big_o_n(n: int):
    k = 0
    for i in range(n):
        k += 1


@time_cost
def big_o_n_2(n: int):
    k = 0
    for i in range(n):
        for j in range(n):
            k += 1


@time_cost
def big_o_n_logn(n: int):
    k = 0
    for i in range(n):
        for _ in [2 ** i for i in range(math.ceil(math.log2(n)))]:
            k += 1


@time_cost
def big_o_n_logn2(n: int):
    k = 0
    for i in range(n):
        j = 1
        while j * 2 < n:
            j *= 2
            k += 1



while True:
    n = int(input("please input n:"))
    which = int(input("please input which algo(1/2/3):"))
    if which == 1:
        # t = timeit.timeit("big_o_n(" + str(n) + ")", "from __main__ import big_o_n", number=1)
        big_o_n(n)
    elif which == 2:
        # t = timeit.timeit("big_o_n_2(" + str(n) + ")", "from __main__ import big_o_n_2", number=1)
        big_o_n_2(n)
    elif which == 3:
        # t = timeit.timeit("big_o_n_logn(" + str(n) + ")", "from __main__ import big_o_n_logn", number=1)
        big_o_n_logn2(n)
    else:
        raise Exception("unknown algo ", which)
    # print(t)
