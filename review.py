#!/usr/bin/env python
# coding=utf-8
from collections import OrderedDict
from typing import Optional, List

import common_data_structure
from common_data_structure import ListNode


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # dp[i][j]表示nums[i...j]无重复子串最大长度
        # dp[i][j] = dp[i-1][j-1] + 2 nums[i] != nums[i + 1]
        #          = max(dp[i][j - 1], dp[i - 1][j])
        # n = len(s)
        # dp = [[0] * n for _ in range(n)]

        n = len(s)
        if n < 1:
            return 0

        ans = 0
        # i = 0
        j = 0
        head = -1
        hmap = dict()
        while j < n:
            if s[j] in hmap:
                head = max(head, hmap[s[j]])
            ans = max(ans, j - head)
            hmap[s[j]] = j
            j += 1
        return ans

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pprev = None
        pcur = head
        while pcur:
            tmp = pcur.next
            pcur.next = pprev
            pprev = pcur
            pcur = tmp

        return pprev

    def findKthLargest(self, nums: List[int], k: int) -> int:

        def partition(nums, left, right):
            pivot = nums[left]
            i = left
            j = right
            while i != j:
                while i < j and pivot >= nums[j]:
                    j -= 1
                while i < j and pivot <= nums[i]:
                    i += 1
                if i < j:
                    nums[i], nums[j] = nums[j], nums[i]

            nums[i], nums[left] = nums[left], nums[i]
            return i

        def quick_select(nums, left, right, target_index):
            q = partition(nums, left, right)
            if q == target_index:
                return nums[q]
            elif q < target_index:
                return quick_select(nums, q + 1, right, target_index)
            else:
                return quick_select(nums, left, q - 1, target_index)

        ans = quick_select(nums, 0, len(nums) - 1, k - 1)
        return ans

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        ans = []
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                i += 1
                continue

            target = - nums[i]
            j = i + 1
            k = n - 1
            while j < k:
                if j > i + 1 and nums[j] == nums[j - 1]:
                    j += 1
                    continue

                if k > j and nums[j] + nums[k] == target:
                    ans.apptail([nums[i], nums[j], nums[k]])
                    k -= 1
                    j += 1
                elif k > j and nums[j] + nums[k] > target:
                    k -= 1
                else:
                    j += 1

        return ans

    def reverse(self, head, tail):
        pcur = head
        prev = tail.next
        # 前节点一直移动至尾节点为止，当前节点移动至尾节点的下一节点(None)
        while prev != tail:
            pnext = pcur.next
            pcur.next = prev
            prev = pcur
            pcur = pnext
        # prev 新翻转链表的头
        # pcur 移动到不翻转节点
        return prev, pcur


    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        def reverse(head, tail):
            p = head
            prev = tail.next
            while prev != tail:
                pnext = p.next
                p.next = prev
                prev = p
                p = pnext
            return tail, head

        dummy = ListNode(-1, head)
        prev = dummy
        start = head
        end = prev
        while start:
            for i in range(k):
                end = end.next
            if end == None:
                break
            nxt = end.next
            start, end = reverse(start, end)
            prev.next = start
            end.next = nxt
            prev = end
            start = nxt
        return dummy.next


s = "abba"
solution = Solution()


# ans = solution.lengthOfLongestSubstring(s)
head = common_data_structure.build_linklist_from_list([1,2,3,4,5])
tail = head
while tail.next:
    tail = tail.next

ans = solution.reverse(head, tail)
# nums = [3,0,-2,-1,1,2]
# ans = solution.threeSum(nums)
print(ans)
print()


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            self.cache[key] = value
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


# Your LRUCache object will be instantiated and called as such:
# ["LRUCache","get","put","get","put","put","get","get"]
# [[2],[2],[2,6],[1],[1,5],[1,2],[1],[2]]
obj = LRUCache(2)
param_1 = obj.get(2)
obj.put(2, 6)
param_2 = obj.get(1)
obj.put(1, 5)
obj.put(1, 2)
param_3 = obj.get(1)
param_4 = obj.get(2)
