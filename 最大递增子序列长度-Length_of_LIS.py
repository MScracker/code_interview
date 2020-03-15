# -*- coding: utf-8 -*- 
# @since : 2020-03-12 12:16 
# @author : wongleon

while True:
    nums = map(int, raw_input().split(','))
    n = len(nums)
    if n == 0:
        break
    dp = [1 for i in range(n)]
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    print dp

