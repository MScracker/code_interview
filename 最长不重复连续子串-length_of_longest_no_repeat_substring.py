# -*- coding: utf-8 -*- 
# @since : 2020-03-13 00:24 
# @author : wongleon

# def lengthOfLongestSubstring(s):
#     """
#     :type s: str
#     :rtype: int
#     """
#     n = len(s)
#     if n == 0:
#         return 0
#     hash_map = {}
#     head = -1
#     max_length = 0
#     for i, c in enumerate(s):
#         if c in hash_map:
#             head = max(head, hash_map[c])
#         hash_map[c] = i
#         max_length = max(max_length, i - head)
#     return max_length

# def lengthOfLongestSubstring(s: str) -> int:
#     from collections import defaultdict
#     max_len = 1
#     dic = defaultdict(int)
#     head = 0
#
#     for tail in range(len(s)):
#         dic[s[tail]] += 1
#
#         while len(dic) < tail - head + 1:
#             dic[s[head]] -= 1
#             if dic[s[head]] == 0:
#                 del dic[s[head]]
#             head += 1
#
#         max_len = max(max_len, tail - head + 1)
#
#     return max_len


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


s = "abcabcbb"
res = lengthOfLongestSubstring(s)
print(res)
