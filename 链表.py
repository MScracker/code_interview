#!/usr/bin/env python
# coding=utf-8
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def removeElements(head: Optional[ListNode], val: int) -> Optional[ListNode]:
    dummy = ListNode(next=head)
    prev = dummy
    cur = prev.next
    while cur:
        if cur.val == val:
            prev.next = cur.next
        else:
            prev = cur
        cur = cur.next

    return dummy.next


l = [1, 2, 7, 4, 5, 7, ]
pcur = ListNode(l[0])
head = pcur

for i in range(1, len(l)):
    pnext = ListNode(l[i])
    pcur.next = pnext
    pcur = pnext

newhead = removeElements(head, 7)

print()
