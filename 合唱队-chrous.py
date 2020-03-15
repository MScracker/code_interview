# coding:utf-8

# def lengthOfLIS(nums):
#     """
#     :type nums: List[int]
#     :rtype: int
#     """
#     n = len(nums)
#     if n == 0:
#         return 0
#     dp = [1 for i in range(n)]
#     max_length = 1
#     for i in range(n):
#         for j in range(i):
#             if(nums[j] < nums[i]):
#                 dp[i] = max(dp[i], dp[j] + 1)
#             max_length = max(max_length, dp[i])
#     return dp
#
# while True:
#     try:
#         n = input()
#         height = map(int, raw_input().split())
#         left = lengthOfLIS(height)
#         right = lengthOfLIS(height[::-1])
#         right.reverse()
#
#         res = 0
#         for i in range(0, n):
#             res = max(res, left[i] + right[i] - 1)
#         print n - res
#
#     except:
#         break

import bisect

while True:
    try:
        n = int(input())
        list1 = list(map(int, raw_input().split()))


        def find_sort_arr(list1, n):
            arr = [99999999] * n
            arr[0] = list1[0]
            res = []
            res += [1]
            for i in range(1, len(list1)):
                pos = bisect.bisect_left(arr, list1[i])
                res.append(pos + 1)
                arr[pos] = list1[i]
            return res


        res1 = find_sort_arr(list1, n)
        res2 = find_sort_arr(list1[::-1], n)[::-1]
        result_arr = list(map(lambda x, y: x + y, res1, res2))
        print(n - (max(result_arr) - 1))
    except:
        break
