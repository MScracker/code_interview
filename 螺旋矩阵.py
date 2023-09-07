#!/usr/bin/env python
# coding=utf-8
from typing import List


def generateMatrix(n: int) -> List[List[int]]:
    cnt = 1
    nums = [[0] * n for _ in range(n)]
    start_x = 0
    start_y = 0
    offsert = 1

    for _ in range(n // 2):
        i = start_x
        j = start_y
        while (j < n - offsert):
            nums[i][j] = cnt
            cnt += 1
            j += 1

        while (i < n - offsert):
            nums[i][j] = cnt
            cnt += 1
            i += 1

        while (j > start_y):
            nums[i][j] = cnt
            cnt += 1
            j -= 1

        while (i > start_x):
            nums[i][j] = cnt
            cnt += 1
            i -= 1

        start_y += 1
        start_x += 1
        offsert += 1

    if n % 2 != 0:
        nums[start_x][start_y] = cnt

    return nums


print(generateMatrix(4))
