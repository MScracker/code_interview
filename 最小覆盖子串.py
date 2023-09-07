#!/usr/bin/env python
# coding=utf-8

def check(source_dic, target_dic):
    for k, v in target_dic.items():
        if k not in source_dic:
            return False
        else:
            if source_dic[k] < target_dic[k]:
                return False
    return True


def minWindow(s: str, t: str) -> str:
    head = 0
    res = ""
    from collections import defaultdict, Counter
    target_dic = Counter(t)
    source_dic = defaultdict(int)

    import sys
    min_len = sys.maxsize
    for tail in range(len(s)):
        source_dic[s[tail]] += 1

        while check(source_dic, target_dic) and head < tail:
            if min_len > tail - head + 1:
                res = s[head: tail + 1]
                min_len = tail - head + 1
            source_dic[s[head]] -= 1
            head += 1

    return res


def minWindow2(s: 'str', t: 'str') -> 'str':
    from collections import defaultdict
    lookup = defaultdict(int)
    for c in t:
        lookup[c] += 1
    start = 0
    end = 0
    min_len = float("inf")
    counter = len(t)  # 所需字母个数
    res = ""
    while end < len(s):
        if lookup[s[end]] > 0:
            counter -= 1
        lookup[s[end]] -= 1

        while counter == 0:  # 表示s[start:end]包含t
            if min_len > end - start + 1:
                min_len = end - start + 1
                res = s[start:end + 1]
            if lookup[s[start]] == 0:  # head再次遇到所需字母，字母个数再次增加准备end下一次左扩
                counter += 1
            lookup[s[start]] += 1
            start += 1
        end += 1
    return res


if __name__ == '__main__':
    s = "ADOBECODEBANC"
    t = "ABC"
    res = minWindow2(s, t)
    print(res)
