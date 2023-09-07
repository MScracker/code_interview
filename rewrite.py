from typing import List, Set


def zero_one_pack_problem_2d(pack_capacity: int, capacity: List[int], value: List[int]):
    item_nums = len(capacity)
    # dp[i][j]表示物品0到物品i都放入容量为j的背包后的最大价值
    dp = [[0] * (pack_capacity + 1) for _ in range(item_nums)]

    # 初始化边界条件，当背包容量为0时
    for i in range(item_nums):
        dp[i][0] = 0
    # 初始化边界条件，第0号物品放入背包时
    for j in range(capacity[0], pack_capacity + 1):
        dp[0][j] = value[0]

    for i in range(1, item_nums):
        for j in range(0, pack_capacity + 1):
            # 第一种情况：dp[i][j] = dp[i - 1][j]表示物品i已经装不进背包容量为j的背包
            # 第二种情况：dp[i][j] = dp[i - 1][j - capacity[i]]表示物品i可以装进背包容量为j的背包
            if j >= capacity[i]:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - capacity[i]] + value[i])
            else:
                # 容量为j的背包装不下物品i
                dp[i][j] = dp[i - 1][j]

    return dp[-1][-1]


def zero_one_pack_problem_1d(pack_capacity: int, capacity: List[int], value: List[int]):
    item_nums = len(capacity)
    # dp[j]表示物品放入容量为j的背包的最大价值
    dp = [0] * (pack_capacity + 1)
    # 初始化边界条件，背包为0时，dp[0]=0
    dp[0] = 0

    for i in range(item_nums):
        # dp是一维数组，为保证每次只能将物品i装入背包中一次，所以必须从后往前遍历
        for j in range(pack_capacity, capacity[i] - 1, -1):
            # 第一种情况：dp[j] = dp[j]表示物品i已经装不进背包容量为j的背包
            # 第二种情况：dp[j] = dp[j - capacity[i]]表示物品i可以装进背包容量为j的背包
            dp[j] = max(dp[j], dp[j - capacity[i]] + value[i])

    return dp[-1]

#物品数量无限多
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


pack_capacity = 4
capacity = [1, 3, 4]
value = [15, 20, 30]


class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1

        for i in range(len(coins)):
            for j in range(0, amount + 1):
                if j >= coins[i]:
                    dp[j] += dp[j - coins[i]]
            print(dp)

        return dp[-1]

    def coinChange(self, coins: List[int], amount: int) -> int:

        # dp[j] = min(dp[j], dp[j - capacity[i]] + value[i])
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] = min(dp[j], dp[j - coins[i]] + 1)
        print(dp)
        return dp[-1]

    def numSquares(self, n: int) -> int:
        dp = [float('inf')] * (n + 1)
        capacity = []

        for i in range(n):
            if i * i <= n:
                capacity.append(i)

        dp[0] = 0
        for i in range(0, n + 1):
            for j in range(i * i, n + 1):
                dp[j] = min(dp[j], dp[j - i * i] + 1)
            print(i, ":", dp)
        print("----", dp)
        return dp[-1]


    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        if s == "":
            return False

        dp = [False] * len(s)
        dp[0] = s[0] in wordDict

        for i in range(len(wordDict)):
            for j in range(len(wordDict[i]), len(s)):
                dp[j] = dp[j] or dp[j - len(wordDict[i])]

        return dp[-1]

solution = Solution()
s= "leetcode"
wordDict = ["leet", "code"]
ans = solution.wordBreak()
print(ans)










