import sys


def minSubArrayLen(target, nums):
    if sum(nums) < target:
        return 0
    min_len = sys.maxsize
    n = len(nums)
    head = 0
    tmp_sum = 0
    for tail in range(n):
        tmp_sum += nums[tail]
        while tmp_sum >= target:
            min_len = min(min_len, tail - head + 1)
            tmp_sum -= nums[head]
            head += 1
    return min_len


nums = [2, 3, 1, 2, 4, 3]
minSubArrayLen(7, nums)
