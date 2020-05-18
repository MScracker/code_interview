# -*- coding: utf-8 -*- 
# @since : 2020/5/8 15:56 
# @author : wongleon


def merge2(nums1, m, nums2, n):
    """
    :type nums1: List[int]
    :type m: int
    :type nums2: List[int]
    :type n: int
    :rtype: None Do not return anything, modify nums1 in-place instead.
    """
    nums1_copy = nums1.copy()

    p1, p2 = 0, 0
    if p1 < m and p2 < n:
        if nums1[p1] < nums2[p2]:
            num1.append(nums1_copy[p1])
            p1 += 1
        else:
            nums2.append(nums2[p2])
            p2 += 1
    if p1 < m:
        num1[p1 + p2:] = nums1_copy[p1:m]
    if p2 < n:
        num1[p1 + p2:] = nums2[p2:n]


def merge(nums1, m, nums2, n):
    """
    :type nums1: List[int]
    :type m: int
    :type nums2: List[int]
    :type n: int
    :rtype: None Do not return anything, modify nums1 in-place instead.
    """
    # two get pointers for nums1 and nums2
    p1 = m - 1
    p2 = n - 1
    # set pointer for nums1
    p = m + n - 1

    # while there are still elements to compare
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] < nums2[p2]:
            nums1[p] = nums2[p2]
            p2 -= 1
        else:
            nums1[p] = nums1[p1]
            p1 -= 1
        p -= 1

    # add missing elements from nums2
    nums1[:p2 + 1] = nums2[:p2 + 1]


num1 = [1, 2, 3, 0, 0, 0]
m = 3
num2 = [2, 5, 6]
n = 3

merge(num1, m, num2, n)
print num1
