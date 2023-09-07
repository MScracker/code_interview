import math
from pprint import pprint
from typing import List, Optional, Set

import common_data_structure
from common_data_structure import TreeNode, build_tree_from_list, pretty_print, ListNode

# 二分法
def search(nums: List[int], target: int) -> int:
   left = 0
   right = len(nums) - 1
   while left <= right:
       mid = (left + right) // 2
       if target > nums[mid]:
           left = mid + 1
       elif target == nums[mid]:
           return mid
       elif target < nums[mid]:
           right = mid - 1
   return -1

# 二分法最左边界
def binary_search_left_insert(nums: [int], target: int):
    left = 0
    right = len(nums) - 1

    while left < right:
        mid = (right + left) // 2
        if target > nums[mid]:
            left = mid + 1  # 寻找区间[mid+1, right]
        elif target == nums[mid]:
            right = mid  # 查找区间[left, mid], 右边界包含目标值
        elif target < nums[mid]:
            right = mid - 1  # 查找区间[left, mid - 1]
    if nums[left] == target:
        return left  # 最左边界的目标值
    return -1

# 二分法最右边界
def binary_search_right_insert(nums: [int], target: int):
    left = 0
    right = len(nums) - 1

    while left < right:
        mid = (left + right + 1) // 2  # 必须向上取整，否则陷入死循环
        if target > nums[mid]:
            left = mid + 1  # 查找区间[mid+1, right]
        elif target == nums[mid]:
            left = mid  # 查找区间[mid, right], 左边界包含目标值
        elif target < nums[mid]:
            right = mid - 1  # 查找区间[left, mid - 1], 包含目标值
    if nums[right] == target:
        return right  # 最右边界的目标值
    return -1


# 两数之和
def two_sum(nums: List[int], target: int) -> List[int]:
    records = dict()
    for index, value in enumerate(nums):
        # 遍历当前元素，并在map中寻找是否有匹配的key
        if target - value in records:
            return [records[target - value], index]
        # 遍历当前元素，并在map中寻找是否有匹配的key
        records[value] = index
    return []

# 三数之和
def threeSum(nums: List[int]) -> List[List[int]]:
    n = len(nums)
    if n <= 3 and sum(nums) != 0:
        return []

    res = []
    # 使目标数组成为有序数组，以解决不重复三元组的问题
    nums = sorted(nums)
    for i in range(n):
        # 有重复元素
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        target = -nums[i]  # 固定目标值
        k = n - 1  # k尾指针，从倒数第一个元素开始
        for j in range(i + 1, n):  # j头指针，必须从i+1搜索
            # [j, k]区间内寻找target
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            while j < k and nums[j] + nums[k] > target:
                k -= 1
                # 头尾指针相遇，表示target
            if j == k:
                break
            # 找到target，记录到结果列表中
            if nums[j] + nums[k] == target:
                res.append([nums[i], nums[j], nums[k]])
    return res

# 四数之和
def four_sum(nums, target):
    freq = {}
    for num in nums:
        freq[num] = freq.get(num, 0) + 1
    ans = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            for k in range(j + 1, len(nums)):
                val = target - (nums[i] + nums[j] + nums[k])
                if val in freq:
                    count = (nums[i] == val) + (nums[j] == val) + (nums[k] == val)
                    if freq[val] > count:
                        ans.add(tuple(sorted([nums[i], nums[j], nums[k], val])))
    return ans

# 摆动序列
def wiggleMaxLength(nums: List[int]) -> int:
    # up i作为波峰最长的序列长度
    # down i作为波谷最长的序列长度
    n = len(nums)
    # 长度为0和1的直接返回长度

    if n < 2: return n
    up = down = 1
    for i in range(1, n):
        if nums[i] > nums[i - 1]:
            # nums[i] 为波峰，1. 前面是波峰，up值不变，2. 前面是波谷，down值加1
            # 目前up值取两者的较大值(其实down+1即可，可以推理前一步down和up最多相差1，所以down+1>=up)
            up = max(up, down + 1)
        elif nums[i] < nums[i - 1]:
            # nums[i] 为波谷，1. 前面是波峰，up+1，2. 前面是波谷，down不变，取较大值
            down = max(down, up + 1)
    return max(up, down)

# 最大子序和
def maxSubArray(nums: List[int]) -> int:
    if not nums:
        return
    # dp[i]表示从0到i的子序列中最大序列和的值
    dp = [0] * len(nums)
    dp[0] = nums[0]
    for i in range(1, len(nums)):
        if dp[i - 1] > 0:
            dp[i] = dp[i - 1] + nums[i]
        else:
            dp[i] = nums[i]
    pprint(dp)
    return max(dp)

# 买卖股票的最佳时机
def maxProfit1(prices: List[int]) -> int:
    low = float("inf")
    result = 0
    for i in range(len(prices)):
        # 取最左最小价格
        low = min(low, prices[i])
        #直接取最大区间利润
        result = max(result, prices[i] - low)
    return result

# 繁琐版：设置成本价
# def maxProfit1(prices: List[int]) -> int:
#
#     profit = 0
#     cost_price = -1
#     for i in range(len(prices)):
#         if prices[i] > cost_price and cost_price >= 0:
#             profit += prices[i] - cost_price
#             cost_price = -1
#
#         if i + 1 < len(prices) and prices[i] < prices[i + 1]:
#             if cost_price < 0:
#                 cost_price = prices[i]
#
#     return profit

# 买卖股票的最佳时机2
def maxProfit2(prices: List[int]) -> int:

    if len(prices) == 0:
        return 0

    # dp[i][0]表示持有股票的所得现金价值
    # dp[i][1]表示不持有股票的最多现金价值
    dp = [[0, 0] for i in range(len(prices))]
    dp[0] = [-prices[0], 0]
    for i in range(1, len(prices)):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i])
        dp[i][1] = max(dp[i - 1][0] + prices[i] - prices[i - 1], dp[i - 1][1])
        print(dp)
    return dp[-1][1]

# 买卖股票的最佳时机
# def maxProfit(prices: List[int]) -> int:
#     i = 0
#     j = len(prices) - 1
#     buy = float('inf')
#     sell = 0
#     while i < j:
#         buy = min(buy, prices[i])
#         sell = max(sell, prices[j])
#         i += 1
#         j -= 1
#
#     if i == j:
#         if prices[i] > sell:
#             sell = prices[i]
#         elif prices[i] < buy:
#             buy = prices[i]
#
#     return max(sell - buy, 0)

# 跳跃游戏2
# 给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。
# 每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处:
# 0 <= j <= nums[i]
# i + j < n
# 返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。
# 示例 1:
#
# 输入: nums = [2,3,1,1,4]
# 输出: 2
# 解释: 跳到最后一个位置的最小跳跃数是 2。
#      从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
# 示例 2:
# 输入: nums = [2,3,0,1,4]
# 输出: 2
# 贪心算法：移动下标只要遇到当前覆盖最远距离的下标，直接步数加一，不考虑是不是终点的情况
#         想要达到这样的效果，只要让移动下标，最大只能移动到 nums.size - 2 的地方就可以了
def jump(nums: List[int]) -> int:

    if len(nums) == 1:
        return 0

    current_coverage = 0 # 当前覆盖的最远距离下标
    next_coverage = 0 # 下一步覆盖的最远距离下标
    i = 0
    step = 0 # 记录走的最少步数
    while i < len(nums) - 1: # 注意这里是小于len(nums) - 1，这是关键所在
        next_coverage = max(i + nums[i], next_coverage) # 更新下一步覆盖的最远距离下标
        if i == current_coverage: # 遇到当前覆盖的最远距离下标
            step += 1
            current_coverage = next_coverage # 更新当前覆盖的最远距离下标

        i = i + 1

    return step

# K次取反后最大化的数组和
def largestSumAfterKNegations(nums: List[int], k: int) -> int:

    # nums.sort()
    # while k > 0:
    #     nums[0] = - nums[0]
    #     nums.sort()
    #     k -= 1
    nums.sort(key=abs, reverse=True)
    pprint(nums)
    for i in range(len(nums)):
        if nums[i] < 0 and k > 0:
            nums[i] *= -1
            k -= 1

    k = k % 2
    if k > 0:
        nums[-1] *= -1

    return sum(nums)

# 加油站 贪心算法
def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
    # 当前累计的剩余油量
    curSum = 0
    # 总剩余油量
    totalSum = 0
    # 起始位置
    start = 0

    for i in range(len(gas)):
        curSum += gas[i] - cost[i]
        totalSum += gas[i] - cost[i]

        # 当前累计剩余油量curSum小于0
        if curSum < 0:
            # 起始位置更新为i+1
            start = i + 1
            # curSum重新从0开始累计
            curSum = 0

    # 总剩余油量totalSum小于0，说明无法环绕一圈
    if totalSum < 0:
        return -1
    return start

# 加油站 贪心算法2
# def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
#     n = len(gas)
#     cur_sum = 0
#     min_sum = float('inf')
#
#     for i in range(n):
#         cur_sum += gas[i] - cost[i]
#         min_sum = min(min_sum, cur_sum)
#
#     if cur_sum < 0: return -1
#     if min_sum >= 0: return 0
#
#     for j in range(n - 1, 0, -1):
#         min_sum += gas[j] - cost[j]
#         if min_sum >= 0:
#             return j
#
#     return -1

# 加油站 暴力解法 超时
# def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
#
#     for i in range(len(gas)):
#         rest = gas[i] - cost[i]
#         start = (i + 1) % len(gas)
#         while start != i and rest >= 0:
#             rest += gas[start] - cost[start]
#             start = (start + 1) % len(gas)
#
#         if rest >= 0 and start == i:
#             return start
#
#     return -1

# 分发糖果
def candy(ratings: List[int]) -> int:

    result = [1] * len(ratings)
    # 从前向后遍历，处理右侧比左侧评分高的情况
    for i in range(1, len(ratings)):
        if ratings[i] > ratings[i - 1]:
            result[i] = result[i - 1] + 1

    i = len(ratings) - 2
    # 从后向前遍历，处理左侧比右侧评分高的情况
    while i >= 0:
        if ratings[i] > ratings[i + 1]:
            result[i] = max(result[i], result[i + 1] + 1)
        i -= 1

    # 统计结果
    return sum(result)

# 根据身高重建队列
def reconstructQueue(people: List[List[int]]) -> List[List[int]]:
    # 先按照h维度的身高顺序从高到低排序。确定第一个维度
    # lambda返回的是一个元组：当-x[0](维度h）相同时，再根据x[1]（维度k）从小到大排序
    people.sort(key=lambda x: (-x[0], x[1]))
    que = []

    # 根据每个元素的第二个维度k，贪心算法，进行插入
    # people已经排序过了：同一高度时k值小的排前面。
    for p in people:
        que.insert(p[1], p)
    return que

# 用最少数量的箭引爆气球
def findMinArrowShots(points: List[List[int]]) -> int:
    points.sort(key=lambda x: x[0])
    arrow = 1
    for i in range(1, len(points)):
        # 更新重叠气球最小右边界
        if points[i][0] <= points[i - 1][1]:
            points[i][1] = min(points[i][1], points[i - 1][1])
        # 气球i和气球i-1不挨着，注意这里不是>=
        else:
            arrow += 1

    return arrow

# 无重叠区间
def eraseOverlapIntervals(intervals: List[List[int]]) -> int:
    # 按照左边界升序排序
    intervals.sort(key=lambda x: x[0])
    # 不重叠区间数量，初始化为1，因为至少有一个不重叠的区间
    cnt = 1
    for i in range(1, len(intervals)):
        # 存在重叠区间
        if intervals[i][0] < intervals[i - 1][1]:
            # 更新重叠区间的右边界
            intervals[i][1] = min(intervals[i][1], intervals[i - 1][1])
        else: # 没有重叠
            cnt += 1

    return len(intervals) - cnt

# 合并区间
def merge(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0])

    ans = []
    start = intervals[0][0]
    end = intervals[0][1]
    for i in range(1, len(intervals)):
        if end >= intervals[i][0]:
            # 区间有重叠，只更新最大右边界
            end = max(intervals[i][1], end)
        else:
            # 无重叠时，将上一次合并过[start, end]加入结果集
            ans.append([start, end])
            start = intervals[i][0]
            end = intervals[i][1]
    # 只有一组或者最后一组合并后[start, end]
    ans.append([start, end])
    return ans

# 单调递增数字
def monotoneIncreasingDigits(n: int) -> int:
    digits = list(str(n))

    # 从右往左遍历字符串
    i = len(digits) - 1
    flag = len(digits)
    while i >= 1:
        # 如果当前字符比前一个字符小，说明需要修改前一个字符
        if digits[i] < digits[i - 1]:
            flag = i
            digits[i - 1] = str(int(digits[i - 1]) - 1)
        i -= 1

    # 将修改位置后面的字符都设置为9，因为修改前一个字符可能破坏了递增性质
    for i in range(flag, len(digits)):
        digits[i] = '9'

    # 将最终的字符串转换回整数并返回
    return int("".join(digits))

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# 监控二叉树
def minCameraCover(root: Optional[TreeNode]) -> int:
    ans = 0

    # 0 : 无覆盖
    # 1 : 有摄像头
    # 2 : 有覆盖
    def traversal(root) -> int:

        nonlocal ans
        if root == None:
            return 2

        left = traversal(root.left)
        right = traversal(root.right)

        # 情况1: 左右节点都有覆盖
        if left == 2 and right == 2:
            return 0

        # 情况2:
        # left == 0 && right == 0 左右节点无覆盖
        # left == 1 && right == 0 左节点有摄像头，右节点无覆盖
        # left == 0 && right == 1 左节点无覆盖，右节点有摄像头
        # left == 0 && right == 2 左节点无覆盖，右节点覆盖
        # left == 2 && right == 0 左节点覆盖，右节点无覆盖
        if left == 0 or right == 0:
            ans += 1
            return 1

        # 情况3:
        # left == 1 && right == 2 左节点有摄像头，右节点有覆盖
        # left == 2 && right == 1 左节点有覆盖，右节点有摄像头
        # left == 1 && right == 1 左右节点都有摄像头
        if left == 1 or right == 1:
            return 2

    if traversal(root) == 0:
        ans += 1

    return ans

# 不同路径/路径和
def uniquePaths(m: int, n: int) -> int:
    # dp[i][j] ：表示从（0 ，0）出发，到(i, j) 有dp[i][j]条不同的路径。
    dp = [[0] * n for _ in range(m)]
    for i in range(m):
        dp[i][0] = 1

    for j in range(n):
        dp[0][j] = 1

    # 计算每个单元格的唯一路径数
    # i,j索引从1开始即可，因为i=0或者j=0都已经初始化好了
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

    # 返回右下角单元格的唯一路径数
    return dp[-1][-1]

# 不同路径2/路径和
def uniquePathsWithObstacles(obstacleGrid: List[List[int]]) -> int:
    m = len(obstacleGrid)
    n = len(obstacleGrid[0])
    # dp[i][j]表示从（0 ，0）出发，到(i, j) 有dp[i][j]条不同的路径。
    dp = [[0] * n for _ in range(m)]

    for i in range(m):
        # 遇到障碍物时，直接退出循环，后面默认都是0
        if obstacleGrid[i][0] != 1:
            dp[i][0] = 1

    for j in range(n):
        if obstacleGrid[0][j] != 1:
            dp[0][j] = 1

    for i in range(1, m):
        for j in range(1, n):
            if obstacleGrid[i][j] != 1:
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

    return dp[m - 1][n - 1]

# 不同的二叉搜索树
def numTrees(n: int) -> int:
    # dp[i]: 1到i为节点组成的二叉搜索树的个数为dp[i]
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n + 1):
        # 对于每个数字i，计算以i为根节点的二叉搜索树的数量
        for j in range(1, i):
            # 利用动态规划的思想，累加左子树和右子树的组合数量
            dp[i] += dp[j - 1] * dp[i - j]

    # 返回以1到n为节点的二叉搜索树的总数量
    return dp[n]

# 分割等和子集
def canPartition(nums: List[int]) -> bool:
    # dp[j] = max(dp[j], dp[j - weight[j] + value[j]])
    # dp[i][j] = max(dp[i-1][j], dp[i-1][j - weight[i]] + value[j])
    target = sum(nums)
    if target % 2 == 1:
        return False
    target //= 2
    # dp[j]表示背包总容量（所能装的总重量）是j，放进物品后，背的最大重量为dp[j]
    dp = [0] * (target + 1)
    for i in range(len(nums)):
        # 从target_sum逆序迭代到num，步长为-1
        for j in range(target, nums[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])

    return target == dp[target]

# 回溯暴力解法
# def canPartition(nums: List[int]) -> bool:
#     n = len(nums)
#     total_sum = sum(nums)
#     path = []
#
#     def backtrace(start):
#         if start >= n - 1:
#             if sum(path) == total_sum / 2:
#                 return True
#             return False
#
#         for i in range(start, n):
#             if sum(path) == total_sum / 2:
#                 return True
#             path.append(nums[i])
#             if backtrace(i + 1):
#                 return True
#             else:
#                 path.pop()
#
#         return False
#
#     return backtrace(0)

# 目标和
# 给你一个非负整数数组 nums 和一个整数 target 。
# 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：
# 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
# 返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。
# 示例 1：
# 输入：nums = [1,1,1,1,1], target = 3
# 输出：5
# 解释：一共有 5 种方法让最终目标和为 3 。
# -1 + 1 + 1 + 1 + 1 = 3
# +1 - 1 + 1 + 1 + 1 = 3
# +1 + 1 - 1 + 1 + 1 = 3
# +1 + 1 + 1 - 1 + 1 = 3
# +1 + 1 + 1 + 1 - 1 = 3
# 示例 2：
# 输入：nums = [1], target = 1
# 输出：1
def findTargetSumWays(nums: List[int], target: int) -> int:
    total_sum = sum(nums)
    value = (target + total_sum)
    # 此时没有方案
    if value % 2 != 0:
        return 0
    if math.fabs(target) > sum(nums):
        return 0

    # 目标和
    value = value // 2
    # dp[j] 表示：填满包含j在内这么大容积的包，有dp[j]种方法
    dp = [0] * (value + 1)
    # 当目标和为0时，只有一种方案，即什么都不选
    dp[0] = 1

    for i in range(len(nums)):
        for j in range(value, nums[i] - 1, -1):
            # 状态转移方程，累加不同选择方式的数量
            dp[j] += dp[j - nums[i]]

    # 返回达到目标和的方案数
    return dp[-1]

# 一和零
def findMaxForm(strs: List[str], m: int, n: int) -> int:
    # dp[i][j]:最多有i个0和j个1的strs的最大子集的大小为dp[i][j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # 遍历物品
    for s in strs:
        zero_num = s.count("0")
        one_num = s.count("1")
        # 遍历背包容量且从后向前遍历
        for i in range(m, zero_num - 1, -1):
            for j in range(n, one_num - 1, -1):
                dp[i][j] = max(dp[i][j], dp[i - zero_num][j - one_num] + 1)

    return dp[-1][-1]

# 打家劫舍
def rob(nums: List[int]) -> int:
    # dp[i]：考虑下标i（包括i）以内的房屋，最多可以偷窃的金额为dp[i]
    # dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
    if len(nums) <= 0:
        return 0

    if len(nums) == 1:
        return nums[0]

    if len(nums) == 2:
        return max(nums)

    dp = [0] * (len(nums))
    # 将dp的第一个元素设置为第一个房屋的金额
    dp[0] = nums[0]
    # 将dp的第二个元素设置为第一二个房屋中的金额较大者
    dp[1] = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        # 对于每个房屋，选择抢劫当前房屋和抢劫前一个房屋的最大金额
        dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])

    return dp[-1]

# 无重复字符的最长子串
def lengthOfLongestSubstring(s: str) -> int:
    max_len = 0
    from collections import defaultdict
    lookup = defaultdict(int) # 当前子串字符及个数
    head = 0 # 头指针
    repeat = 0  # 重复字符总个数
    for tail in range(len(s)):
        # 初始化lookup，更新条件repeat标志位
        lookup[s[tail]] += 1
        if lookup[s[tail]] > 1:
            # 1个及以上即存在重复字符
            repeat += 1
        # repeat不满足条件：s[head,tail]有重复字符
        # 头指针尽可能保守右移，一旦满足条件就跳出循环
        while repeat > 0:
            if lookup[s[head]] > 1:
                # 只有当s[head]数量达到两个以上时，确保跳出循环时保证s[head, tail]满足无重复字符
                repeat -= 1
            # 头指针尽右移并更新lookup
            lookup[s[head]] -= 1
            head += 1
        # while循环外更新不重复子串最大长度
        max_len = max(max_len, tail - head + 1)
    return max_len


# 最长递增子序列
def lengthOfLIS(nums: List[int]) -> int:
    # dp[i]表示以nums[i]结尾的最长递增子序列的长度
    # dp[i] = dp[i - 1] + 1, nums[i] > nums[i - 1]
    # dp[i] = dp[i - 1]
    dp = [1] * len(nums)
    result = 0
    for i in range(1, len(nums)):
        for j in range(0, i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
        result = max(result, dp[i])

    return result

# 最长递增子序列的个数
def findNumberOfLIS(nums: List[int]) -> int:
    # dp[i]表示以nums[i]及其之前元素结尾的最长递增子序列的长度
    # count[i]表示nums[i]结尾的最长递增子序列的个数
    # dp[i] = max(dp[i], dp[j] + 1)  nums[j] < nums[i]
    n = len(nums)
    dp = [1] * n
    count = [1] * n
    max_length = 1
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                if dp[i] == dp[j] + 1:
                    count[i] += count[j]
                elif dp[i] < dp[j] + 1:
                    count[i] = count[j]
                    dp[i] = dp[j] + 1
                max_length = max(max_length, dp[i])
    ans = 0
    for i in range(n):
        if max_length == dp[i]:
            ans += count[i]
    return ans


# 最长重复子数组
# 动态边界问题简洁版
def findLength(num1: List[int], num2: List[int]) -> int:
    # dp[i][j] 表示nums1[i-1]结尾和nums2[j-1]结尾的最长重复子数组的长度
    dp = [[0] * (len(num2) + 1) for _ in range(len(num1) + 1)]
    ans = 0
    for i in range(1, len(num1) + 1):
        for j in range(1, len(num2) + 1):
            if num1[i - 1] == num2[j - 1]:
                dp[i][j] = dp[i-1][j-1] + 1
            ans = max(ans, dp[i][j])
    return ans

# 动态边界问题繁琐版
# def findLength(nums1: List[int], nums2: List[int]) -> int:
#     # dp[i][j] 表示nums1[i]结尾和nums2[j]结尾的最长重复子数组的长度
#     # dp[i][j] = dp[i - 1][j] + 1  #nums1[i] == nums2[j]
#     # dp[i][j] = dp[i][j - 1] + 1
#     if len(nums1) == 0 or len(nums2) == 0:
#         return 0
#
#     dp = [[0] * len(nums2) for _ in range(len(nums1))]
#     ans = 0
#     for i in range(len(nums1)):
#         if nums1[i] == nums2[0]:
#             dp[i][0] = 1
#         ans = max(ans, dp[i][0])
#
#     for j in range(len(nums2)):
#         if nums1[0] == nums2[j]:
#             dp[0][j] = 1
#         ans = max(ans, dp[0][j])
#
#     for i in range(1, len(nums1)):
#         for j in range(1, len(nums2)):
#             if nums1[i] == nums2[j]:
#                 dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1)
#             ans = max(ans, dp[i][j])
#     return ans

# 最长公共子序列
# 动态规划简洁版
def longestCommonSubsequence(text1: str, text2: str) -> int:
    len1, len2 = len(text1) + 1, len(text2) + 1
    # dp[i][j]：长度为[0, i - 1]的字符串text1与长度为[0, j - 1]的字符串text2的最长公共子序列为dp[i][j]
    dp = [[0 for _ in range(len1)] for _ in range(len2)]
    for i in range(1, len2):
        for j in range(1, len1):
            # 如果text1[i - 1] 与 text2[j - 1]相同，那么找到了一个公共元素
            if text1[j - 1] == text2[i - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            # 如果text1[i - 1] 与 text2[j - 1]不相同，
            # 那就看text1[0, i - 2]与text2[0, j - 1]的最长公共子序列
            # 和 text1[0, i - 1]与text2[0, j - 2]的最长公共子序列，取最大的
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]

# 动态规划繁琐版
# def longestCommonSubsequence(text1: str, text2: str) -> int:
#     #dp[i][j]：长度为[0, i]的字符串text1与长度为[0, j]的字符串text2的最长公共子序列为dp[i][j]
#     dp = [[0] * len(text2) for _ in range(len(text1))]
#     for i in range(len(text1)):
#         if text1[i] == text2[0]:
#             dp[i][0] = 1
#
#     for j in range(len(text2)):
#         if text1[0] == text2[j]:
#             dp[0][j] = 1
#
#     ans = 0
#     for i in range(len(text1)):
#         for j in range(len(text2)):
#             if text1[i] != text2[j]:
#                 if i == 0 and j == 0:
#                     continue
#                 if j == 0:
#                     dp[i][j] = dp[i - 1][j]
#                     continue
#                 if i == 0:
#                     dp[i][j] = dp[i][j - 1]
#                     continue
#
#                 dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
#
#             elif text1[i] == text2[j] and i > 0 and j > 0:
#                 dp[i][j] = dp[i - 1][j - 1] + 1
#             ans = max(ans, dp[i][j])
#
#     return ans

# 不相交的线: 求绘制的最大连线数，其实就是求两个字符串的最长公共子序列的长度
def maxUncrossedLines(nums1: List[int], nums2: List[int]) -> int:

    n1 = len(nums1)
    n2 = len(nums2)
    dp = [[0] * n2 for _ in range(n1)]

    for i in range(n1):
        if nums1[i] == nums2[0]:
            dp[i][0] = 1

    for j in range(n2):
        if nums1[0] == nums2[j]:
            dp[0][j] = 1
    ans = 0
    for i in range(n1):
        for j in range(n2):
            if nums1[i] == nums2[j]:
                if i > 0 and j > 0:
                    dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                if i == 0 and j == 0:
                    continue
                elif i > 0 and j > 0:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
                elif i == 0:
                    dp[i][j] = dp[i][j - 1]
                elif j == 0:
                    dp[i][j] = dp[i - 1][j]

            ans = max(ans, dp[i][j])

    return ans

# 判断子序列
# 法一：遍历法
def isSubsequence(s: str, t: str) -> bool:
    ps = 0
    pt = 0
    while ps < len(s) and pt < len(t):
        if s[ps] == t[pt]:
            ps += 1
            pt += 1
        else:
            pt += 1

    if ps == len(s):
        return True
    else:
        return False

# 法二：动态规划
def isSubsequence(s: str, t: str) -> bool:
    if s == "":
        return True

    if t == "" or len(t) < len(s):
        return False
    # dp[i][j] 表示以下标i-1为结尾的字符串s，和以下标j-1为结尾的字符串t，相同子序列的长度为dp[i][j]
    dp = [[0] * (len(t) + 1) for _ in range(len(s) + 1)]

    for i in range(1, len(s) + 1):
        for j in range(1, len(t) + 1):
            # t中找到了一个字符在s中也出现了
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            # 相当于t要删除元素，继续匹配
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        print(dp)
    return dp[-1][-1] == len(s)

# 编辑距离
# dp[i][j] 表示以下标i-1为结尾的字符串word1，和以下标j-1为结尾的字符串word2，最近编辑距离为dp[i][j]。
def minDistance(word1: str, word2: str) -> int:
    # dp[i][j]表示以word1[i - 1]结尾的字符最少需要删除几个字符变成以word2[j - 1]结尾的字符
    # word1[i] == word2[j] dp[i][j] = dp[i - 1][j - 1]
    # word1[i] != word2[j] dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    n1 = len(word1)
    n2 = len(word2)
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

    for i in range(n1 + 1):
        dp[i][0] = i

    for j in range(n2 + 1):
        dp[0][j] = j

    dp[0][0] = 0

    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    return dp[-1][-1]

# 回文子串
def countSubstrings(s: str) -> int:
    # dp[i][j]：表示区间范围[i,j] （注意是左闭右闭）的子串是否是回文子串
    dp = [[False] * len(s) for _ in range(len(s))]

    for i in range(len(s)):
        dp[i][0] = False

    for j in range(len(s)):
        dp[0][j] = s[:j + 1] == s[:j + 1][::-1]

    ans = 0
    for i in range(len(s) - 1, -1, -1):
        for j in range(i, len(s)):
            if s[i] == s[j]:
                # 情况一：下标i与j相同，同一个字符例如a，当然是回文子串
                # 情况二：下标i与j相差为1，例如aa，也是回文子串
                if j - i <= 1:
                    dp[i][j] =True
                else:
                    # 情况三：下标i与j相差大于1的时候，例如cabac，此时s[i]与s[j]已经相同了，我们看i到j区间是不是回文子串是不是回文就可以
                    dp[i][j] = dp[i + 1][j - 1]

            else:
                dp[i][j] = False

            if dp[i][j]:
                ans += 1
    return ans

# 最长回文子序列
def longestPalindromeSubseq(s: str) -> int:
    n = len(s)
    # dp[i][j]：字符串s在[i, j]范围内最长的回文子序列的长度为dp[i][j]
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                dp[i][j] = 1

    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if s[i] == s[j]:
                # 如果s[i]与s[j]相同，那么dp[i][j] = dp[i + 1][j - 1] + 2
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                # 如果s[i]与s[j]不相同，说明s[i]和s[j]的同时加入并不能增加[i,j]区间回文子序列的长度
                # 那么分别加入s[i]、s[j]看看哪一个可以组成最长的回文子序列:
                # 加入s[j]的回文子序列长度为dp[i + 1][j]。
                # 加入s[i]的回文子序列长度为dp[i][j - 1]。
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

    return dp[0][-1]


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 链表相交
# 双指针法
# 当链表headA和headB都不为空时，创建两个头指针pA和pB，将两个指针依次遍历两个链表的每个节点，其中一个遍历完了就遍历另一个链表，直至指向同一节点或null节点为止
def getIntersectionNode(headA: ListNode, headB: ListNode) -> Optional[ListNode]:
    if not headA or not headB:
        return None

    pa = headA
    pb = headB
    while pa and pb:
        if pa == pb:
            return pa

        if pa.next is None and pb.next is None:
            return None

        if pa.next:
            pa = pa.next
        else:
            pa = headB

        if pb.next:
            pb = pb.next
        else:
            pb = headA

    return None

# 反转链表
def reverseBetween(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    dummy = ListNode(-1, next=head)
    p = dummy
    prev = None
    right_node = None

    for _ in range(left - 1):
        p = p.next
        prev = p

    for _ in range(right - left + 1):
        p = p.next
        right_node = p

    left_node = prev.next
    succ = right_node.next

    prev.next = None
    right_node.next = None

    def reverse(head):
        prev = None
        p = head
        while p:
            tmp = p.next
            p.next = prev
            prev = p
            p = tmp

    reverse(left_node)
    prev.next = right_node
    left_node.next = succ
    return head

# 每日温度
# 单调栈：就是用一个栈来记录我们遍历过的元素, 并保证栈内元素自底向上单调递增
def dailyTemperatures(temperatures: List[int]) -> List[int]:
    ans = [0] * len(temperatures)

    from collections import deque
    stack = deque()

    for i in range(len(temperatures)):
        if len(stack) == 0 or temperatures[i] < temperatures[stack[-1]]:
            stack.append(i)
        else:
            # 如果当前遍历的元素大于栈顶元素，表示栈顶元素的右边的最大的元素就是当前遍历的元素，
            # 所以弹出栈顶元素并记录
            while len(stack) > 0 and temperatures[i] >= temperatures[stack[-1]]:
                idx = stack.pop()
                ans[idx] = i - idx

            stack.append(i)

    return ans



# 接雨水
# 双指针法
def trap(height: List[int]) -> int:
    ans = 0
    # 记录每个柱子左边柱子最大高度
    max_left = [0] * len(height)
    max_left[0] = height[0]
    # 记录每个柱子右边柱子最大高度
    max_right = [0] * len(height)
    max_right[-1] = height[-1]
    for i in range(1, len(height)):
        max_left[i] = max(height[i], max_left[i - 1])
    for i in range(len(height) - 2, -1, -1):
        max_right[i] = max(height[i], max_right[i + 1])

    for i in range(len(height)):
        ans += min(max_left[i], max_right[i]) - height[i]

    return ans

# 单调栈压缩版
# 单调栈是按照 行 的方向来计算雨水
# 从栈顶到栈底的顺序：从小到大
# 通过三个元素来接水：栈顶，栈顶的下一个元素，以及即将入栈的元素
# 雨水高度是 min(凹槽左边高度, 凹槽右边高度) - 凹槽底部高度
# 雨水的宽度是 凹槽右边的下标 - 凹槽左边的下标 - 1（因为只求中间宽度）
def trap(height: List[int]) -> int:
    stack = [0]
    result = 0
    for i in range(1, len(height)):
        while stack and height[i] > height[stack[-1]]:
            mid_height = stack.pop()
            if stack:
                # 雨水高度是 min(凹槽左侧高度, 凹槽右侧高度) - 凹槽底部高度
                h = min(height[stack[-1]], height[i]) - height[mid_height]
                # 雨水宽度是 凹槽右侧的下标 - 凹槽左侧的下标 - 1
                w = i - stack[-1] - 1
                # 累计总雨水体积
                result += h * w
        stack.append(i)
    return result


# 柱状图中最大的矩形
def largestRectangleArea(heights: List[int]) -> int:
    # 两个DP数列储存的均是下标index
    # 记录每个柱子的左侧第一个矮一级的柱子的下标
    # 初始化-1，防止while死循环
    min_left_index = [-1] * len(heights)
    # 记录每个柱子的右侧第一个矮一级的柱子的下标
    # 初始化柱子个数防止while死循环
    min_right_index = [len(heights)] * len(heights)

    for i in range(1, len(heights)):
        # 以当前柱子为主心骨，向左迭代寻找次级柱子
        t = i - 1
        while t >= 0 and heights[t] >= heights[i]:
            # 当左侧的柱子持续较高时，尝试这个高柱子自己的次级柱子（DP
            t = min_left_index[t]
        # 当找到左侧矮一级的目标柱子时
        min_left_index[i] = t

    for i in range(len(heights) - 2, -1, -1):
        # 以当前柱子为主心骨，向右迭代寻找次级柱子
        t = i + 1
        while t < len(heights) and heights[t] >= heights[i]:
            # 当右侧的柱子持续较高时，尝试这个高柱子自己的次级柱子
            t = min_right_index[t]
        # 当找到右侧矮一级的目标柱子时
        min_right_index[i] = t

    ans = 0
    for i in range(len(heights)):
        cur = (min_right_index[i] - min_left_index[i] - 1) * heights[i]
        ans = max(ans, cur)

    return ans

# 单调栈精简版
# 找每个柱子左右侧的第一个高度值小于该柱子的柱子
# 单调栈：栈顶到栈底：从大到小（每插入一个新的小数值时，都要弹出先前的大数值）
# 栈顶，栈顶的下一个元素，即将入栈的元素：这三个元素组成了最大面积的高度和宽度
# 情况一：当前遍历的元素heights[i]大于栈顶元素的情况
# 情况二：当前遍历的元素heights[i]等于栈顶元素的情况
# 情况三：当前遍历的元素heights[i]小于栈顶元素的情况
def largestRectangleArea(heights: List[int]) -> int:

    heights.insert(0, 0)
    heights.append(0)
    stack = [0]
    result = 0
    for i in range(1, len(heights)):
        while stack and heights[i] < heights[stack[-1]]:
            mid_height = heights[stack[-1]]
            stack.pop()
            if stack:
                # area = width * height
                area = (i - stack[-1] - 1) * mid_height
                result = max(area, result)
        stack.append(i)
    return result

# 最长回文串
# 中心扩散法
def longestPalindrome(s: str) -> str:
    def center_expand(s, left, right):
        while left <= right and left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1

        return left + 1, right - 1

    maxlen = 0
    start = 0
    end = 0
    for i in range(len(s) - 1):
        start1, end1 = center_expand(s, i, i)
        start2, end2 = center_expand(s, i, i + 1)

        if maxlen < end1 - start1:
            maxlen = end1 - start1
            start = start1
            end = end1

        if maxlen < end2 - start2:
            maxlen = end2 - start2
            start = start2
            end = end2

    return s[start: end + 1]

# 动态规划法
def longestPalindrome2(s: str) -> str:
    # dp[i][j]表示s[i...j]是否为回文串
    n = len(s)
    max_length = 1
    ans = s[0]

    dp = [[False] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = True

    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if s[i] != s[j]:
                dp[i][j] = False
            else:
                if i + 1 < j - 1:
                    dp[i][j] = dp[i + 1][j - 1]
                else:
                    dp[i][j] = True

            if dp[i][j] and j - i + 1 > max_length:
                max_length = j - i + 1
                ans = s[i: j + 1]
    return ans

# 分割回文串1
def partition(s: str) -> List[List[str]]:
    result = []

    def backtracking(s, start_index, path, result):
        # Base Case
        if start_index == len(s):
            result.append(path[:])
            return

        # 单层递归逻辑
        for i in range(start_index, len(s)):
            # 若反序和正序相同，意味着这是回文串
            if s[start_index: i + 1] == s[start_index: i + 1][::-1]:
                path.append(s[start_index:i + 1])
                # 递归纵向遍历：从下一处进行切割，判断其余是否仍为回文串
                backtracking(s, i + 1, path, result)
                # 回溯
                path.pop()

    backtracking(s, 0, [], result)
    return result


# 分割回文串2
def minCut(s: str) -> int:

    # dp[i]表示s[i]结尾的字符串的最少分割次数
    # dp[i] = min(dp[i], dp[j] + 1)
    n = len(s)
    # 递推公式求min，所以必须初始化成理论上最多分割方式即全部分割成单字符
    dp = list(range(n))
    palindrome = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j]:
                if i + 1 >= j - 1:
                    palindrome[i][j] = True
                elif palindrome[i + 1][j - 1]:
                    palindrome[i][j] = True

    for i in range(n):
        if palindrome[0][i]:
            dp[i] = 0
            continue

        for j in range(i):
            if palindrome[j + 1][i]:
                dp[i] = min(dp[i], dp[j] + 1)

    return dp[-1]

# 快速排序
def sortArray(nums: List[int]) -> List[int]:

    def parition(nums, left, right):
        pivot = nums[right]
        i = left
        j = right
        while i != j:
            while i < j and nums[i] <= pivot:
                i += 1

            while i < j and nums[j] >= pivot:
                j -= 1

            if i < j:
                nums[i], nums[j] = nums[j], nums[i]

        nums[right] = nums[i]
        nums[i] = pivot
        return i

    def quick_select(nums, left, right):
        if left >= right:
            return
        pos = parition(nums, left, right)
        quick_select(nums, left, pos - 1)
        quick_select(nums, pos + 1, right)

    n = len(nums)
    quick_select(nums, 0, n - 1)
    return nums

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# 二叉树层序遍历
# 利用队列先进先出暂存每层节点
def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    from collections import deque
    que = deque()
    ans = []

    if not root:
        return ans

    que.append(root)
    while que:
        n = len(que)
        res = []
        for i in range(n):
            ele = que.popleft()
            res.append(ele.val)
            if ele.left:
                que.append(ele.left)
            if ele.right:
                que.append(ele.right)
        ans.append(res)

    return ans

# 二叉树中的最大路径和
def maxPathSum(root: Optional[TreeNode]) -> int:
    max_sum = float('-inf')

    def max_node_sum(root):
        nonlocal max_sum
        if not root:
            return 0

        # 当节点最大贡献值为负的时候，就置为零
        left_gain = max(max_node_sum(root.left), 0)
        right_gain = max(max_node_sum(root.right), 0)
        # 计算当前节点值+左右节点最大贡献值即为当前最大路径和
        path_sum = root.val + left_gain + right_gain
        max_sum = max(max_sum, path_sum)
        return root.val + max(left_gain, right_gain)

    max_node_sum(root)
    return max_sum

# 循环数组
def mock_cycle(nums):
    n = len(nums)
    for i in range(n):
        j = i % n  # 从当前元素开始循环打印
        while j != (i - 1 + n) % n:  # 不等于nums[i]的前一元素
            print("%d " % nums[j], end="")
            j = (j + 1) % n  # [0, 1, ..., n-1]
        print("%d " % nums[j])  # 跳出循环j = (i - 1 + n) % n, 打印nums[i]的前一元素

def mock_cycle2(nums):
    n = len(nums)
    for i in range(n * 2):  # 两倍循环变量，等同于两相同数组拼接在一起
        print("%d " % nums[i % n], end="")


#01背包问题
def zero_one_pack_problem(pack_capacity:int, capacity:List[int], value:List[int]):
    item_nums = len(capacity)
    # dp[j]表示物品放入容量为j的背包的最大价值
    dp = [0] * (pack_capacity + 1)
    # 初始化边界条件，背包为0时，dp[0]=0
    dp[0] = 0
    # 首先遍历物品再遍历背包
    for i in range(item_nums):
        # dp是一维数组，为保证每次只能将物品i装入背包中一次，所以必须从后往前遍历
        for j in range(pack_capacity, capacity[i] - 1, -1):
            # 第一种情况：dp[j] = dp[j]表示物品i已经装不进背包容量为j的背包
            # 第二种情况：dp[j] = dp[j - capacity[i]]表示物品i可以装进背包容量为j的背包
            dp[j] = max(dp[j], dp[j - capacity[i]] + value[i])
    return dp[-1]

#完全背包
def intact_pack_problem_1d(pack_capacity: int, capacity: Set[int], value: Set[int]):
    item_nums = len(capacity)
    # dp[j]表示物品放入容量为j的背包的最大价值
    dp = [0] * (pack_capacity + 1)
    # 初始化边界条件，背包为0时，dp[0]=0
    dp[0] = 0

    for i in range(item_nums):
        # dp是一维数组，从前往后遍历，保证每个物品可以重复取直至背包容量耗尽为止
        for j in range(capacity[i], pack_capacity + 1):
            # 第一种情况：dp[j] = dp[j]表示物品i已经装不进背包容量为j的背包
            # 第二种情况：dp[j] = dp[j - capacity[i]]表示物品i可以装进背包容量为j的背包
            dp[j] = max(dp[j], dp[j - capacity[i]] + value[i])

    return dp[-1]


a = [1,2,3,4,5]
b = [1,5]
headA = common_data_structure.build_linklist_from_list(a)
headB = common_data_structure.build_linklist_from_list(b)
nums = [4,10,4,3,8,9]
ans = lengthOfLIS(nums)
pprint(ans)
