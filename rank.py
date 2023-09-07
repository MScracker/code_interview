#!/usr/bin/env python
# coding=utf-8
import random
from typing import List, Optional

import numpy as np

import common_data_structure
from common_data_structure import TreeNode


class Solution(object):

    def bubble_sort(self, nums):
        n = len(nums)
        # 两两比较n-1轮
        for i in range(n - 1):
            # 每一轮可确保i个元素已经排定
            for j in range(1, n - i):
                # 比较相邻元素，使得最大的元素移动最右侧
                if nums[j] < nums[j - 1]:
                    nums[j], nums[j - 1] = nums[j - 1], nums[j]
        return nums

    def insert_sort(self, nums):
        n = len(nums)
        # 比较n-1轮
        # nums[0...i)保存前i个有序元素
        for i in range(1, n):
            for j in range(i, 0, -1):
                if nums[j] < nums[j - 1]:
                    nums[j], nums[j - 1] = nums[j - 1], nums[j]
                else:
                    break
        return nums

    def merge_sort(self, nums):
        def merge(left, mid, right, temp):
            for i in range(left, right + 1):
                temp[i] = nums[i]
            i = left
            j = mid + 1
            for k in range(left, right + 1):
                if i == mid + 1:
                    nums[k] = temp[j]
                    j += 1
                elif j == right + 1:
                    nums[k] = temp[i]
                    i += 1
                elif temp[i] <= temp[j]:
                    nums[k] = temp[i]
                    i += 1
                else:
                    nums[k] = temp[j]
                    j += 1

        def partition(nums, left, right, temp):
            if left == right:
                return

            mid = (left + right) // 2
            # 分
            partition(nums, left, mid, temp)
            partition(nums, mid + 1, right, temp)
            # 合
            merge(left, mid, right, temp)

        temp = [0] * len(nums)
        partition(nums, 0, len(nums) - 1, temp)
        return nums

    def isValid(self, s: str) -> bool:
        n = len(s)
        if n < 2 or n % 2 != 0:
            return False

        from collections import defaultdict
        dic = defaultdict(list)
        for i in range(n):
            if s[i] == '(' or s[i] == '[' or s[i] == '{':
                dic[s[i]] += [i]
            else:
                if s[i] == ')':
                    if len(dic['(']) == 0:
                        return False
                    elif dic['('] and dic['('].pop() > i:
                        return False

                elif s[i] == ']':
                    if len(dic['[']) == 0:
                        return False
                    elif dic['['] and dic['['].pop() > i:
                        return False
                elif s[i] == '}':
                    if len(dic['{']) == 0:
                        return False
                    elif dic['{'] and dic['{'].pop() > i:
                        return False

        return True

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i = m - 1
        j = n - 1
        k = m + n - 1
        while k >= 0:
            if j < 0:
                nums1[k] = nums1[i]
                i -= 1
            elif i < 0:
                nums1[k] = nums2[j]
                j -= 1
            elif nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1

            k -= 1

    def numIslands(self, grid: List[List[str]]) -> int:
        ans = 0
        num_row = len(grid)
        num_col = len(grid[0])

        def dfs(grid, r, c):
            if not 0 <= r < num_row or not 0 <= c < num_col:
                return

            if grid[r][c] != '1':
                return

            grid[r][c] = '2'
            # 上
            dfs(grid, r - 1, c)
            # 下
            dfs(grid, r + 1, c)
            # 左
            dfs(grid, r, c - 1)
            # 右
            dfs(grid, r, c + 1)

        for r in range(num_row):
            for c in range(num_col):
                if grid[r][c] == '1':
                    ans += 1
                    dfs(grid, r, c)

        return ans

    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        ans = []
        from collections import deque
        res = deque()
        if not root:
            return
        res.append(root)
        depth = 0
        while res:
            size = len(res)
            tmp = deque()
            for i in range(size):
                node = res.popleft()
                if depth % 2 == 0:
                    tmp.append(node.val)
                else:
                    tmp.appendleft(node.val)
                if node.left:
                    res.append(node.left)
                if node.right:
                    res.append(node.right)
            ans.append(list(tmp))
            depth += 1
        return ans

    def permute(self, nums: List[int]) -> List[List[int]]:

        n = len(nums)
        ans = []
        path = []
        used = [False] * n

        def backtrace(nums, used):
            if len(path) == n:
                ans.append(path[:])
                return

            for i in range(n):
                if used[i]:
                    continue
                path.append(nums[i])
                used[i] = True
                backtrace(nums, used)
                used[i] = False
                path.pop()

        backtrace(nums, used)
        return ans

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m = len(matrix)
        n = len(matrix[0])
        ans = []

        left = 0
        right = n - 1
        top = 0
        bottom = m - 1
        num = m * n
        while left <= right and top <= bottom:

            i = top
            j = left
            while j < right and num > 0:
                ans.append(matrix[i][j])
                j += 1
                num -= 1

            i = top
            while i < bottom and num > 0:
                ans.append(matrix[i][j])
                i += 1
                num -= 1

            j = right
            while j > left and num > 0:
                ans.append(matrix[i][j])
                j -= 1
                num -= 1

            i = bottom
            while i > top and num > 0:
                ans.append(matrix[i][j])
                i -= 1
                num -= 1

            top += 1
            bottom -= 1
            left += 1
            right -= 1

        if m == n and m % 2 == 1:
            ans.append(matrix[m // 2][m // 2])

        return ans

    def generateMatrix(self, n: int) -> [[int]]:
        l, r, t, b = 0, n - 1, 0, n - 1
        mat = [[0 for _ in range(n)] for _ in range(n)]
        num, tar = 1, n * n
        while num <= tar:
            for i in range(l, r + 1):  # left to right
                mat[t][i] = num
                num += 1
            t += 1
            for i in range(t, b + 1):  # top to bottom
                mat[i][r] = num
                num += 1
            r -= 1
            for i in range(r, l - 1, -1):  # right to left
                mat[b][i] = num
                num += 1
            b -= 1
            for i in range(b, t - 1, -1):  # bottom to top
                mat[i][l] = num
                num += 1
            l += 1
        return mat

matrix = [[6,9,7]] #np.arange(1, 17).reshape(4, 4).tolist() #[[2, 3, 6]]  #
# [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
# [1,2,3,4,8,12,11,10,9,5,6,7]
solution = Solution()
# ans = solution.spiralOrder(matrix)
ans = solution.generateMatrix(4)
print(ans)
