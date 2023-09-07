# -*- coding: utf-8 -*- 
# @since : 2020/4/21 21:51 
# @author : wongleon
from ChainList import ChainList
from common_data_structure import ListNode


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy_head = ListNode(-1)
        cur = dummy_head
        carry = 0
        while l1 or l2:
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            sum = x + y + carry
            carry = sum / 10
            cur.next = ListNode(sum % 10)
            cur = cur.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        if carry > 0:
            cur.next = ListNode(carry)

        return dummy_head.next


l1 = ChainList()
l1.append(1)
l2 = ChainList()
l2.append(9)
l2.append(9)
solution = Solution()
h = solution.addTwoNumbers(l1.head, l2.head)
print h
