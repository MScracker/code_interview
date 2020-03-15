# -*- coding: utf-8 -*- 
# @since : 2020-03-14 23:26 
# @author : wongleon

import math
import traceback
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    for i in xrange(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def find(i):
    for j in xrange(len(odds)):
        if links[i][j] == 1 and used[j] == 0: #links[i][j]==1表示两个索引代表的值和能否凑成素数 used[j]==0表示试图为j更改匹配关系没有成功
            used[j] = 1
            if matches[j] == -1 or find(matches[j]): #-1表示没匹配上 or matches[j]一定有值匹配上，递归腾位置
                matches[j] = i
                return True
    return False

while True:
    try:
        n = input()
        nums = map(int, raw_input().split())
        odds = filter(lambda x : x%2 == 1, nums)
        evens = filter (lambda x : x%2 == 0, nums)
        links = [[0 for j in xrange(len(odds))] for i in xrange(len(evens))]
        for i in xrange(len(evens)):
            for j in xrange(len(odds)):
                if is_prime(evens[i] + odds[j]):
                    links[i][j] = 1
        matches = [-1 for j in xrange(len(odds))] #匹配关系数组
        count = 0
        for i in xrange(len(evens)):
            used = [0 for j in xrange(len(odds))] #本轮匹配关系，每轮清零
            if find(i):
                count += 1
        print count
    except:
        traceback.print_exc()
        break