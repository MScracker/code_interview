# -*- coding: utf-8 -*- 
# @since : 2020-03-13 00:24 
# @author : wongleon

def lengthOfLongestSubstring(s):
    """
    :type s: str
    :rtype: int
    """
    n = len(s)
    if n == 0:
        return 0
    hash_map = {}
    head = -1
    max_length = 0
    for i, c in enumerate(s):
        if c in hash_map:
            head = max(head, hash_map[c])
        hash_map[c] = i
        max_length = max(max_length, i - head)
    return max_length

s = 'abba'
lengthOfLongestSubstring(s)