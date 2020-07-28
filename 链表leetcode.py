# -*- coding: utf-8 -*- 
# @since : 2020/6/5 17:07 
# @author : wongleon
from common_data_structure import *

l = [10, 9, 8, 7, 6]

head = build_linklist_from_list(l)

num = 0;
def last_k_node(head, k):
    global num
    num = k
    if not head:
        return

    cur = last_k_node(head.next, k)
    if cur:
        return cur
    else:
        num -= 1
        if num == 0:
            return head


def last_k_node_dual_pointer(head, k):
    if not head:
        return
    p1 = head
    p2 = head
    while k > 0:
        p2 = p2.next
        k -= 1

    while p2:
        p2 = p2.next
        p1 = p1.next
    return p1

def kthToLast(head, k):
    """
    :type head: ListNode
    :type k: int
    :rtype: int
    """
    if not head:
        return
    global NUM
    NUM = k
    def helper(head, k):
        if not head:
            return None
        global NUM
        p = helper(head.next, k) #回溯各节点
        if p is None: #末节点为None或未找回目标节点
            NUM -= 1
            if NUM == 0:
                return head #返回目标节点
        else:
            return p #返回目标节点

    cur = helper(head, k)
    return cur.val

x = kthToLast(head, 2)
print x


def deleteNode(head, val):
    """
    :type head: ListNode
    :type val: int
    :rtype: ListNode
    """
    if head.val == val:
        return head.next
    p = head
    while p.next:
        if p.next.val == val:
            p.next = p.next.next
            break
        p = p.next
    return head

head = build_linklist_from_list([4, 5, 1, 9])

def reversePrintLink(head):
    '''
    逆序打印
    :param head: the head of link list
    :return:
    '''
    if not head:
        return

    reversePrintLink(head.next)
    print str(head.val) + "->",



def reverseLink(head):
    '''
    翻转链表
    :param head:
    :return:
    '''
    if not head or not head.next:
        return head
    tail = reverseLink(head.next)
    head.next.next = head #head为尾节点
    head.next = None
    return tail

new = reverseLink(head)
print new

def deleteNode(head, val):
    """
    :type head: ListNode
    :type val: int
    :rtype: ListNode
    """
    if not head:
        return

    pcur = head
    pnext = head.next
    while pnext:
        if pcur.val == val:
            pcur.val = pnext.val
            pcur.next = pnext.next
            break
        pcur = pcur.next
        pnext = pnext.next
    if not pnext:
        pcur = head
        while pcur.next.val != val:
            pcur = pcur.next
        pcur.next = None