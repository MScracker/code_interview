from typing import List


def sortedSquares(nums: List[int]) -> List[int]:
    n = len(nums)
    result = [0] * n
    i = 0
    j = n - 1
    k = n - 1
    while i < j:
        if nums[i] ** 2 < nums[j] ** 2:
            result[k] = nums[j] ** 2
            j -= 1
            k -= 1
        else:
            result[k] = nums[i] ** 2
            i += 1
            k -= 1
    result[0] = nums[i] ** 2
    return result


nums = [-4, -1, 0, 3, 10]
print(sortedSquares(nums))


while j < len(nums):
    #判断[i, j]是否满足条件
    while 不满足条件：
        i += 1 # 最保守的压缩i，一旦满足条件了就退出压缩i的过程，使得滑窗尽可能的大

    #一旦满足条件，不断更新结果（注意在while外更新！）
    res = max(res) #更新结果
    j += 1 #右移右边界

