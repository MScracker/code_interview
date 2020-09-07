# -*- coding: utf-8 -*- 
# @since : 2020/7/31 18:04 
# @author : wongleon

from common_data_structure import *


def removeZeroSumSublists(head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    sdict = {}
    # dummy = ListNode(0)
    # dummy.next = head

    s = 0
    p = head
    while p:
        s += p.val
        sdict[s] = p
        p = p.next

    s = 0
    p = head
    while p:
        s += p.val
        p.next = sdict[s].next
        p = p.next

    return head


head = build_linklist_from_list([1, 2, -3, 3, 1])


# x = removeZeroSumSublists(head)
# print x

def sortedListToBST(head):
    """
    :type head: ListNode
    :rtype: TreeNode
    """
    if not head:
        return

    def helper(head, tree=None):
        if tree is None:
            tree = TreeNode(head.val)
        tree.left = helper(head.next, tree.left)
        tree.right = helper(head.next.next, tree.right)
        return tree

    return helper(head)


# t = sortedListToBST(head)
# print t

def only_search(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """

    def binary(nums, low, high):
        if low > high:
            return -1
        mid = (low + high) // 2
        if target == nums[mid]:
            return mid
        elif target < nums[mid]:
            return binary(nums, low, mid)
        else:
            return binary(nums, mid + 1, high)

    return binary(nums, 0, len(nums) - 1)


def left_bound(nums, target):
    def binary(nums, low, high):
        mid = (low + high) // 2
        if low == high:
            if nums[mid] == target:
                return mid
            else:
                return -1
        else:
            if nums[mid] >= target:
                return binary(nums, low, mid)
            else:
                return binary(nums, mid + 1, high)

    return binary(nums, 0, len(nums) - 1)



a = [5, 5, 5, 5, 5, 10, 14, 15, 16, 19, 19, 19]
# a = [-1,0,3,5,9,12]
pos = left_bound(a, 19)
print pos


def twisted_binary_search(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """

    l = 0
    r = len(nums) - 1
    while l <= r:

        mid = (l + r) // 2

        if nums[mid] == target:
            return mid
        elif nums[0] <= nums[mid]:
            if nums[l] <= target and target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        else:
            if nums[mid] < target and target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
    return -1


nums = [4, 5, 6, 7, 0, 1, 2]
target = 2
pos = twisted_binary_search(nums, target)
print pos


def checkSubarraySum(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: bool
    """

    n = len(nums)
    if n <= 1:
        return False
    ssum = 0
    sset = {}
    for i in range(n):
        ssum += nums[i]
        if k != 0:
            if ssum % k not in sset.keys():
                sset[ssum % k] = i
            elif i - sset[ssum % k] > 1:
                return True
        else:
            if ssum == 0 and 0 not in sset.keys():
                sset[0] = i
            elif ssum == 0 and 0 in sset.keys() and sset[0] - i >= 1:
                return True
    return False

nums = [0,0]
k = -1
print checkSubarraySum(nums, k)


def subarraySum(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    n = len(nums)
    count = 0
    pre = 0
    d = {0: 1}
    for i in range(n):
        pre += nums[i]
        if pre - k in d:
            count += d[pre - k]
        if pre in d:
            d[pre] += 1
        else:
            d[pre] = 1
    return count

nums = [3,4,7,2,-3,1,4,2]
count  = subarraySum(nums, 7)
print count